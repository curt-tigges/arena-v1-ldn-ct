import torch as t
from torch import nn
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
from IPython.display import display
import pandas as pd
import numpy as np
import transformers
from fancy_einsum import einsum
from dataclasses import dataclass
from tqdm.notebook import tqdm_notebook
import matplotlib

from einops import rearrange, reduce, repeat

import sys 
sys.path.append('../common_modules')

from transformer_modules import Dropout, LayerNorm, MLP, TransformerConfig, Embedding, GELU
from general_modules import Linear

from typing import Optional


class BERTMultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.query_size = int(hidden_size / num_heads)
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.ff = Linear(hidden_size, hidden_size)

    def multihead_masked_attention(self, Q: t.Tensor, K: t.Tensor, V: t.Tensor, additive_attention_mask: Optional[t.Tensor], num_heads: int):
        """
        Implements multihead masked attention on the matrices Q, K and V.

        Q: shape (batch, seq, nheads*headsize)
        K: shape (batch, seq, nheads*headsize)
        V: shape (batch, seq, nheads*headsize)

        returns: shape (batch, seq, nheads*headsize)
        """
        Q = rearrange(Q, "B S (nheads headsize) -> B S nheads headsize", nheads=num_heads)
        K = rearrange(K, "B S (nheads headsize) -> B S nheads headsize", nheads=num_heads)
        V = rearrange(V, "B S (nheads headsize) -> B S nheads headsize", nheads=num_heads)

        batch_size, seq_len, nheads, headsize = Q.shape
        scores = einsum("B Qseq nheads headsize, B Kseq nheads headsize -> B nheads Qseq Kseq", Q, K)
        scores /= Q.shape[-1] ** 0.5

        if additive_attention_mask is not None:
            attention_scores = attention_scores + additive_attention_mask

        scores = t.softmax(scores, dim=-1)
        Z = einsum("B nheads Qseq Kseq, B Kseq nheads headsize -> B Qseq nheads headsize", scores, V)
        Z = rearrange(Z, "B Qseq nheads headsize -> B Qseq (nheads headsize)")
        return Z

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor]) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Z = self.multihead_masked_attention(Q, K, V, additive_attention_mask, self.num_heads)
        out = self.ff(Z)
        return out 


class BERTBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = BERTMultiheadMaskedAttention(config.hidden_size, config.num_heads)
        self.lnorm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config.hidden_size, config.dropout)
        self.lnorm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        additive_attention_mask: shape (batch, nheads=1, seqQ=1, seqK)
        '''
        attn = self.attn(x, additive_attention_mask)
        out = self.lnorm1(attn + x)
        mlp = self.mlp(out)
        out = self.lnorm2(mlp + out)
        return out


def make_additive_attention_mask(one_zero_attention_mask: t.Tensor, big_negative_number: float = -10000) -> t.Tensor:
    '''
    one_zero_attention_mask: 
        shape (batch, seq)
        Contains 1 if this is a valid token and 0 if it is a padding token.

    big_negative_number:
        Any negative number large enough in magnitude that exp(big_negative_number) is 0.0 for the floating point precision used.

    Out: 
        shape (batch, nheads=1, seqQ=1, seqK)
        Contains 0 if attention is allowed, big_negative_number if not.
    '''
    mask = 1 - one_zero_attention_mask
    mask = big_negative_number * mask
    return repeat(mask, 'B S -> B 1 1 S')
    

class BertCommon(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.tkn_emb = nn.Embedding(2, config.hidden_size)

        self.lnorm = LayerNorm(config.hidden_size)
        self.dropout = Dropout(p=config.dropout)

        decoders = [BERTBlock(config) for l in range(config.num_layers)]
        self.blocks = nn.ModuleList(decoders)
        

    def forward(
        self,
        x: t.Tensor,
        one_zero_attention_mask: Optional[t.Tensor] = None,
        token_type_ids: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        '''
        input_ids: (batch, seq) - the token ids
        one_zero_attention_mask: (batch, seq) - only used in training, passed to `make_additive_attention_mask` and used in the attention blocks.
        token_type_ids: (batch, seq) - only used for NSP, passed to token type embedding.
        '''
        # Embeddings
        pos = t.arange(x.shape[1], device=x.device)
        if not token_type_ids:
            token_type_ids = t.zeros_like(x)

        embedding = self.emb(x.long()) + self.pos_emb(pos.long()) + self.tkn_emb(token_type_ids.long())
        #print(embedding.device)
        # Norm & Dropout
        out = self.lnorm(embedding)
        out = self.dropout(out)
        
        # Mask
        if one_zero_attention_mask:
            mask = make_additive_attention_mask(one_zero_attention_mask).to(x.device)
        else:
            mask = None

        for b in self.blocks:
            out = b(out, mask)

        return out

class BERTLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.common = BertCommon(config)
        self.linear = Linear(config.hidden_size, config.hidden_size)
        self.gelu = GELU()
        self.lnorm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.tied_embed_bias = nn.Parameter(t.zeros(config.vocab_size))
    
    def forward(self, x):
        out = self.common(x)
        out = self.gelu(self.linear(out))
        out = self.lnorm(out)
        out = einsum("B S E, V E -> B S V", out, self.common.emb.weight)

        return out


from typing import List


def predict(model, tokenizer, text: str, k=15) -> List[List[str]]:
    '''
    Return a list of k strings for each [MASK] in the input.
    '''
    """
    Return a list of k strings for each [MASK] in the input.
    """
    model.eval()
    tokens = tokenizer.encode(text=text, return_tensors="pt")
    res = model(tokens)
    
    mask_predictions = []
    for n, input_id in enumerate(tokens.squeeze()):
        if input_id == tokenizer.mask_token_id:
            logits = res[0, n]
            top_logits_indices = t.topk(logits, k).indices
            predictions = tokenizer.decode(top_logits_indices)
            mask_predictions.append(predictions)
    
    return mask_predictions

def test_bert_prediction(predict, model, tokenizer):
    '''Your Bert should know some names of American presidents.'''
    text = "Former President of the United States of America, George[MASK][MASK]"
    predictions = predict(model, tokenizer, text)
    print(f"Prompt: {text}")
    print("Model predicted: \n", "\n".join(map(str, predictions)))
    assert "Washington" in predictions[0]
    assert "Bush" in predictions[0]