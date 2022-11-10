import numpy as np
import torch as t
from torch import nn


class Embedding(nn.Module):
    """Returns an embedding of input tokens"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embed = num_embeddings
        self.embed_dim = embedding_dim
        self.weight = nn.Parameter(
            t.ones(num_embeddings, embedding_dim).uniform_(-1, to=1)
        )

    def forward(self, x: t.LongTensor) -> t.Tensor:
        """For each integer in the input, return that row of the embedding."""
        return self.weight[x]

    def extra_repr(self) -> str:
        return f"{self.num_embed}, {self.embed_dim}"


class PositionalEncoding(nn.Module):
    """Adds sin-cosine positional encoding to the input"""

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embedding_dim
        self.n = 10000

        freqs = np.outer(
            np.arange(max_seq_len),
            1 / self.n ** (2 * np.arange(embedding_dim // 2) / embedding_dim),
        )
        enc_2d = np.zeros((max_seq_len, embedding_dim))
        enc_2d[:, ::2] = np.sin(freqs)
        enc_2d[:, 1::2] = np.cos(freqs)
        self.pos_enc = t.from_numpy(enc_2d)
        self.register_buffer("pos_enc", self.pos_enc)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq_len, embedding_dim)
        """
        return x + self.pos_enc[: x.shape[1], :]

    def extra_repr(self) -> str:
        return f"max_freq={self.n}, max_seq_len={self.max_seq_len}, embedding_dim={self.embed_dim}"


class LayerNorm(nn.Module):
    """Performs normalization over specified dimensions"""

    def __init__(
        self, normalized_shape, eps: float = 1e-05, elementwise_affine: bool = True
    ):
        super().__init__()
        self.norm_shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else normalized_shape
        )
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(t.ones(normalized_shape))
            self.bias = nn.Parameter(t.zeros(normalized_shape))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Normalize along each embedding"""
        x_dims, norm_shape_dims = len(x.shape), len(self.norm_shape)
        norm_dims = tuple([d for d in range(x_dims - norm_shape_dims, x_dims)])

        self.mean = t.mean(x, dim=norm_dims, keepdim=True)
        self.var = t.var(x, dim=norm_dims, unbiased=False, keepdim=True)

        out = (x - self.mean) / t.sqrt(self.var + self.eps)

        if self.elementwise_affine:
            out = out * self.weight + self.bias

        return out

    def extra_repr(self) -> str:
        return f"normalized_shape={self.norm_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class Dropout(nn.Module):
    """Returns activations to which the Dropout technique has been applied"""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.training:
            d_shape = x.shape
            dropout_matrix = t.rand(d_shape)
            dropout_matrix[dropout_matrix < self.p] = 0
            dropout_matrix[dropout_matrix >= self.p] = 1
            # should this be on the device?
            out = x * dropout_matrix
            out = out / (1 - self.p)
            return out
        else:
            return x

    def extra_repr(self) -> str:
        return f"p={self.p}"


class GELU(nn.Module):
    """Performs the GELU approximation"""

    def forward(self, x: t.Tensor) -> t.Tensor:
        out = (
            x * 0.5 * (1 + t.tanh(t.sqrt(t.tensor(2 / t.pi)) * (x + 0.044715 * x**3)))
        )
        return out
