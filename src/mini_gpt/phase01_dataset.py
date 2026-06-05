"""
Phase 1: Dataset Creation
===========================

Turn raw text into (input, target) tensor pairs for next-token prediction.

Theory (see docs/phase01_dataset_creation.md):
    - Tokenize text → 1D sequence of token IDs
    - Slide a context window of length T over the sequence
    - Input  x = tokens[i : i+T]
    - Target y = tokens[i+1 : i+T+1]   (shifted by +1)

Tensor shapes:
    token_ids : (N,)     N = total tokens in corpus
    x (input) : (B, T)   B = batch size, T = context window
    y (target): (B, T)   same shape as x; y[b,i] is the token AFTER x[b,i]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def get_tokenizer(name: str = "gpt2") -> tiktoken.Encoding:
    """
    Return a pre-trained BPE tokenizer.

    Why tiktoken?
        We are allowed to use an existing tokenizer (project rule #10).
        GPT-2 BPE splits text into subword tokens and maps each to an integer ID.

    Vocabulary size V ≈ 50,257 for gpt2.
    """
    return tiktoken.get_encoding(name)


def load_and_tokenize(
    file_path: str | Path,
    encoding_name: str = "gpt2",
) -> torch.Tensor:
    """
    Read a text file and convert it to a 1D tensor of token IDs.

    Steps:
        1. Read raw text (string)
        2. Encode with BPE tokenizer → list of integers
        3. Wrap in torch.LongTensor

    Returns:
        token_ids : shape (N,)
            - Each element is one token ID (scalar integer)
            - Row-wise it's a single sequence; think of it as one long column
              of N tokens read left-to-right through the corpus

    Example:
        "I love AI" → [40, 1848, 9552]  (actual IDs depend on tokenizer)
    """
    file_path = Path(file_path)
    # Read entire file as one string
    text: str = file_path.read_text(encoding="utf-8")

    # Encode: string → list[int], one int per BPE token
    tokenizer = get_tokenizer(encoding_name)
    ids: list[int] = tokenizer.encode(text)

    # LongTensor required for nn.Embedding (Phase 2)
    token_ids = torch.tensor(ids, dtype=torch.long)

    # DEBUG: always print shape when learning
    print(f"[load_and_tokenize] text length = {len(text)} chars")
    print(f"[load_and_tokenize] token_ids.shape = {tuple(token_ids.shape)}  (N,)")

    return token_ids


# ---------------------------------------------------------------------------
# Input / Target construction
# ---------------------------------------------------------------------------

def build_input_target_pairs(
    token_ids: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split a 1D token sequence into aligned input and target matrices.

    Math:
        For corpus s = (s_0, s_1, ..., s_{N-1}) and block size T:

        num_blocks = N - T   (one block per valid start index)

        x[b, t] = s_{b + t}        input token at position t in block b
        y[b, t] = s_{b + t + 1}    target = next token (shift by +1)

    Args:
        token_ids  : (N,) all tokens in corpus
        block_size   : T, context window length

    Returns:
        x : (num_blocks, T)  input tokens
        y : (num_blocks, T)  target tokens (shifted +1)

    Rows    = independent training examples (different sliding windows)
    Columns = time steps within one example (token positions 0..T-1)
    """
    N = token_ids.shape[0]  # total number of tokens
    T = block_size

    if N <= T:
        raise ValueError(
            f"Corpus has N={N} tokens but block_size T={T}. "
            f"Need N > T to create at least one training example."
        )

    # num_blocks = how many starting positions fit in the corpus
    num_blocks = N - T

    # Pre-allocate matrices
    # x and y both have shape (num_blocks, T)
    x = torch.zeros(num_blocks, T, dtype=torch.long)
    y = torch.zeros(num_blocks, T, dtype=torch.long)

    # Fill each row: one sliding window
    for b in range(num_blocks):
        # x[b] = tokens[b : b+T]       → T consecutive input tokens
        x[b] = token_ids[b : b + T]
        # y[b] = tokens[b+1 : b+T+1]   → same window shifted right by 1
        y[b] = token_ids[b + 1 : b + T + 1]

    print(f"[build_input_target_pairs] N={N}, T={T}, num_blocks={num_blocks}")
    print(f"[build_input_target_pairs] x.shape = {tuple(x.shape)}  (B, T)")
    print(f"[build_input_target_pairs] y.shape = {tuple(y.shape)}  (B, T)")

    return x, y


def verify_shift(
    x: torch.Tensor,
    y: torch.Tensor,
    token_ids: torch.Tensor,
    block_size: int,
) -> None:
    """
    Sanity check: y[b, t] must equal the token immediately after x[b, t]
    in the original corpus.

    For block b starting at index b in token_ids:
        x[b, t] = token_ids[b + t]
        y[b, t] = token_ids[b + t + 1]
    """
    B, T = x.shape
    for b in range(min(B, 3)):  # check first 3 rows only (debug)
        for t in range(T):
            expected = token_ids[b + t + 1].item()
            actual = y[b, t].item()
            assert actual == expected, (
                f"Shift error at row={b}, col={t}: "
                f"expected y={expected}, got {actual}"
            )
    print(f"[verify_shift] OK — y[b,t] == token_ids[b+t+1] for sample rows")


# ---------------------------------------------------------------------------
# PyTorch Dataset & DataLoader
# ---------------------------------------------------------------------------

class TextTokenDataset(Dataset):
    """
    PyTorch Dataset yielding (input, target) pairs for language modeling.

    Each __getitem__ returns:
        x : (T,)  one row of input token IDs
        y : (T,)  one row of target token IDs

    After batching in DataLoader:
        x : (B, T)
        y : (B, T)
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        assert x.shape == y.shape, "input and target must have same shape"
        self.x = x  # (num_examples, T)
        self.y = y  # (num_examples, T)

    def __len__(self) -> int:
        # Number of training examples = number of rows
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Return one (input, target) pair; each shape (T,)
        return self.x[idx], self.y[idx]


def create_dataloader(
    dataset: TextTokenDataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Wrap dataset in a DataLoader for batched training.

    shuffle=True shuffles ROWS (different examples), never columns within a row.
    Shuffling tokens inside a sequence would destroy word order — never do that.

    Returns batches:
        x : (B, T)
        y : (B, T)
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    print(
        f"[create_dataloader] batch_size={batch_size}, "
        f"num_batches={len(loader)}, shuffle={shuffle}"
    )
    return loader


def train_val_split(
    dataset: TextTokenDataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[TextTokenDataset, TextTokenDataset]:
    """
    Split dataset into train and validation subsets.

    val_fraction=0.1 → 90% train, 10% validation.
    """
    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    print(f"[train_val_split] train={n_train}, val={n_val}")
    return train_set, val_set  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Demo — run this file directly to see shapes printed step-by-step
# ---------------------------------------------------------------------------

def demo() -> None:
    """
    Interactive walkthrough of Phase 1.

    Run: python -m src.mini_gpt.phase01_dataset
    """
    print("=" * 60)
    print("PHASE 1 DEMO: Dataset Creation")
    print("=" * 60)

    # Locate sample data relative to project root
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "sample.txt"

    # --- Step 1: Tokenize ---
    print("\n--- Step 1: Tokenize raw text ---")
    token_ids = load_and_tokenize(data_path)
    tokenizer = get_tokenizer()

    # Show first 10 tokens decoded back to text
    sample_ids = token_ids[:10].tolist()
    print(f"First 10 token IDs: {sample_ids}")
    print(f"Decoded: {tokenizer.decode(sample_ids)!r}")

    # --- Step 2: Build input/target pairs ---
    print("\n--- Step 2: Build input/target pairs ---")
    T = 8  # small window for demo visibility
    x, y = build_input_target_pairs(token_ids, block_size=T)
    verify_shift(x, y, token_ids, block_size=T)

    # --- Step 3: Show the "I love → love AI" style example ---
    print("\n--- Step 3: Numerical example (first block) ---")
    print(f"x[0] token IDs: {x[0].tolist()}")
    print(f"y[0] token IDs: {y[0].tolist()}")
    # print(f"x token IDs: {x.tolist()}")
    # print(f"y token IDs: {y.tolist()}")
    print(f"x[0] decoded:   {[tokenizer.decode([t.item()]) for t in x[0]]}")
    print(f"y[0] decoded:   {[tokenizer.decode([t.item()]) for t in y[0]]}")
    print("\nAt each column i: model sees x[0,i], must predict y[0,i] = next token")

    # --- Step 4: Dataset & DataLoader ---
    print("\n--- Step 4: Dataset & DataLoader ---")
    dataset = TextTokenDataset(x, y)
    loader = create_dataloader(dataset, batch_size=4, shuffle=True)

    for batch_idx, (bx, by) in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  bx.shape = {tuple(bx.shape)}  (B, T) — B={bx.shape[0]} sequences, T={bx.shape[1]} positions")
        print(f"  by.shape = {tuple(by.shape)}  (B, T) — same layout, shifted targets")
        if batch_idx <= 1:
            print(f"  bx[0] decoded: {tokenizer.decode(bx[0].tolist())!r}")
            print(f"  by[0] decoded: {tokenizer.decode(by[0].tolist())!r}")
        if batch_idx >= 1:
            break  # only show first 2 batches in demo

    # --- Step 5: Train/val split ---
    print("\n--- Step 5: Train / validation split ---")
    train_set, val_set = train_val_split(dataset, val_fraction=0.1)
    print(f"Train examples: {len(train_set)}, Val examples: {len(val_set)}")

    print("\n" + "=" * 60)
    print("Phase 1 complete. See docs/phase01/theory.md for theory.")
    print("Next: Phase 2 → python -m src.mini_gpt.phase02_embeddings")
    print("=" * 60)


if __name__ == "__main__":
    demo()
