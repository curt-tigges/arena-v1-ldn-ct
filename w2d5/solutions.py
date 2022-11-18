import hashlib
import os
import zipfile
import torch as t
import transformers
from einops import rearrange
from torch.nn import functional as F
from tqdm import tqdm
import requests
import utils

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

MAIN = __name__ == "__main__"
DATA_FOLDER = "./data"
DATASET = "2"
BASE_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/"
DATASETS = {"103": "wikitext-103-raw-v1.zip", "2": "wikitext-2-raw-v1.zip"}
TOKENS_FILENAME = os.path.join(DATA_FOLDER, f"wikitext_tokens_{DATASET}.pt")

if MAIN:
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)

# %%

def maybe_download(url: str, path: str) -> None:
    """Download the file from url and save it to path. If path already exists, do nothing."""
    if not os.path.exists(path):
        with open(path, "wb") as file:
            data = requests.get(url).content
            file.write(data)

# %%

if MAIN:
    path = os.path.join(DATA_FOLDER, DATASETS[DATASET])

    maybe_download(BASE_URL + DATASETS[DATASET], path)
    expected_hexdigest = {"103": "0ca3512bd7a238be4a63ce7b434f8935", "2": "f407a2d53283fc4a49bcff21bc5f3770"}
    with open(path, "rb") as f:
        actual_hexdigest = hashlib.md5(f.read()).hexdigest()
        assert actual_hexdigest == expected_hexdigest[DATASET]

    print(f"Using dataset WikiText-{DATASET} - options are 2 and 103")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

    z = zipfile.ZipFile(path)

    def decompress(*splits: str) -> str:
        return [
            z.read(f"wikitext-{DATASET}-raw/wiki.{split}.raw").decode("utf-8").splitlines()
            for split in splits
        ]

    train_text, val_text, test_text = decompress("train", "valid", "test")
# %%

import functools
def concat_lists(list_of_lists):
    return functools.reduce(lambda x, y: x+y, list_of_lists)

def tokenize_1d(tokenizer, lines: list[str], max_seq: int) -> t.Tensor:
    """Tokenize text and rearrange into chunks of the maximum length.

    Return (batch, seq) and an integer dtype.
    """

    lines_tokenized = tokenizer(
        lines, 
        truncation=False, 
        add_special_tokens=False, 
        padding=False,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    input_ids = lines_tokenized["input_ids"]
    input_ids = concat_lists(input_ids)

    n_to_truncate = len(input_ids) % max_seq
    input_ids = t.tensor(input_ids[:-n_to_truncate]).to(t.int)

    input_ids = rearrange(input_ids, "(b s) -> b s", s=max_seq)

    return input_ids


def tokenize_1d_with_progress_bar(tokenizer, lines: list[str], max_seq: int, n_intervals: int) -> t.Tensor:
    input_ids = []
    interval_len = len(lines) // (n_intervals - 1)
    slices = [slice(i*interval_len, (i+1)*interval_len) for i in range(n_intervals)]
    progress_bar = tqdm(slices)
    for slice_ in progress_bar:
        lines_tokenized = tokenizer(
            lines[slice_], 
            truncation=False, 
            add_special_tokens=False, 
            padding=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        input_ids.append(concat_lists(lines_tokenized["input_ids"]))

    input_ids = concat_lists(input_ids)
    n_to_truncate = len(input_ids) % max_seq
    input_ids = t.tensor(input_ids[:-n_to_truncate]).to(t.int)

    input_ids = rearrange(input_ids, "(b s) -> b s", s=max_seq)

    return input_ids

if MAIN:
    max_seq = 128
    print("Tokenizing training text...")
    train_data = tokenize_1d(tokenizer, train_text, max_seq, 100)
    print("Training data shape is: ", train_data.shape)
    print("Tokenizing validation text...")
    val_data = tokenize_1d(tokenizer, val_text, max_seq)
    print("Tokenizing test text...")
    test_data = tokenize_1d(tokenizer, test_text, max_seq)
    print("Saving tokens to: ", TOKENS_FILENAME)
    t.save((train_data, val_data, test_data), TOKENS_FILENAME)

# %%

def random_mask(
    input_ids: t.Tensor, mask_token_id: int, vocab_size: int, select_frac=0.15, mask_frac=0.8, random_frac=0.1
) -> tuple[t.Tensor, t.Tensor]:
    '''Given a batch of tokens, return a copy with tokens replaced according to Section 3.1 of the paper.

    input_ids: (batch, seq)

    Return: (model_input, was_selected) where:

    model_input: (batch, seq) - a new Tensor with the replacements made, suitable for passing to the BertLanguageModel. Don't modify the original tensor!

    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise
    '''

    input_ids_modified = input_ids.clone()

    # Create masks
    mask_seed = t.randperm(input_ids.numel()).reshape(input_ids.shape).to(input_ids.device)

    threshold_probabilities = t.tensor([
        0,
        select_frac * mask_frac,
        select_frac * (mask_frac + random_frac),
        select_frac
    ])
    threshold_values = input_ids.numel() * threshold_probabilities

    fill_values = [mask_token_id, input_ids.clone().random_(vocab_size)]
    for threshold_lower, threshold_higher, fill_value in zip(threshold_values[0:2], threshold_values[1:3], fill_values):
        input_ids_modified = t.where(
            (threshold_lower <= mask_seed) & (mask_seed < threshold_higher),
            fill_value,
            input_ids_modified
        )

    return input_ids_modified, mask_seed < threshold_values[-1]

if MAIN:
    utils.test_random_mask(random_mask, input_size=10000, max_seq=max_seq)


# %%

if MAIN:
    # Find the word frequencies
    word_frequencies = t.bincount(train_data.flatten())
    # Drop the words with occurrence zero (because these contribute zero to cross entropy)
    word_frequencies = word_frequencies[word_frequencies > 0]
    # Get probabilities
    word_probabilities = word_frequencies / word_frequencies.sum()
    # Calculate the cross entropy
    cross_entropy = (- word_probabilities * word_probabilities.log()).sum()
    print(cross_entropy)
    # ==> 7.3446


# %%

def flat(x: t.Tensor) -> t.Tensor:
    """Combines batch and sequence dimensions."""
    return rearrange(x, "b s ... -> (b s) ...")

def cross_entropy_selected(pred: t.Tensor, target: t.Tensor, was_selected: t.Tensor) -> t.Tensor:
    """
    pred: (batch, seq, vocab_size) - predictions from the model
    target: (batch, seq, ) - the original (not masked) input ids
    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise

    Out: the mean loss per predicted token
    """
    target = t.where(was_selected.to(t.bool), target, -100)
    entropy = F.cross_entropy(flat(pred), flat(target))
    return entropy

if MAIN:
    utils.test_cross_entropy_selected(cross_entropy_selected)

    batch_size = 8
    seq_length = 512
    batch = t.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    pred = t.rand((batch_size, seq_length, tokenizer.vocab_size))
    (masked, was_selected) = random_mask(batch, tokenizer.mask_token_id, tokenizer.vocab_size)
    loss = cross_entropy_selected(pred, batch, was_selected).item()
    print(f"Random MLM loss on random tokens - does this make sense? {loss:.2f}")