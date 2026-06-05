"""
Phase 2: Token Embeddings
==========================

Convert integer token IDs (B, T) into continuous vectors (B, T, C).

Theory (see docs/phase02/theory.md):
    - Embedding matrix W_E has shape (V, C)
    - Each ROW = one vocabulary token's learned vector
    - Each COLUMN = one feature dimension
    - Lookup: token_id k → row k of W_E

Tensor shapes:
    token_ids        : (B, T)      integers — indices into vocabulary
    embedding matrix : (V, C)      learnable weights
    token embeddings : (B, T, C)   float vectors fed to transformer
"""

from __future__ import annotations

from pathlib import Path

import tiktoken
import torch
import torch.nn as nn

from src.mini_gpt.phase01_dataset import (
    build_input_target_pairs,
    create_dataloader,
    load_and_tokenize,
    TextTokenDataset,
)


# Default Mini GPT hyperparameter (Phase 11 will use same value)
DEFAULT_EMBED_DIM = 128


def get_vocab_size(encoding_name: str = "gpt2") -> int:
    """
    Return vocabulary size V for a tiktoken encoding.

    GPT-2: V = 50,257
    """
    enc = tiktoken.get_encoding(encoding_name)
    return enc.n_vocab


class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer — turns token IDs into dense vectors.

    This is a LOOKUP TABLE, not a matrix multiply in the usual sense.

    Math:
        Given token ID k at position (b, t):
            output[b, t, :] = W_E[k, :]     (row k of embedding matrix)

    Shapes:
        W_E (self.weight) : (V, C)
            Rows    = vocabulary tokens (token ID 0 .. V-1)
            Columns = embedding dimensions (C features per token)

        Input  token_ids : (B, T)
        Output embeddings : (B, T, C)
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size  # V
        self.embed_dim = embed_dim    # C

        # W_E: learnable embedding matrix, shape (V, C)
        # nn.Parameter tells PyTorch: "train this with gradient descent"
        # Small random init (std=0.02) — standard for GPT-style models
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim) * 0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embedding vector for each token ID.

        Args:
            token_ids : (B, T) LongTensor of integers in [0, V-1]

        Returns:
            embeddings : (B, T, C) FloatTensor
                Row b    = batch example b
                Column t = token position t in the sequence
                Last dim = C-dimensional embedding vector at (b, t)
        """
        # Safety checks (helpful while learning)
        assert token_ids.dtype == torch.long, "token_ids must be torch.long (integers)"
        assert token_ids.dim() == 2, f"expected (B, T), got shape {token_ids.shape}"

        # LOOKUP: for each token_id, fetch the corresponding ROW from self.weight
        # token_ids (B,T) indexes into weight (V,C) → result (B,T,C)
        embeddings = self.weight[token_ids]
        return embeddings

    def get_row(self, token_id: int) -> torch.Tensor:
        """
        Fetch the embedding vector for a single token ID.

        Useful for debugging: "what vector does token 40 ('I') get?"

        Returns:
            vector shape (C,)
        """
        return self.weight[token_id]


def embed_batch(
    token_ids: torch.Tensor,
    embedding_layer: TokenEmbedding,
) -> torch.Tensor:
    """
    Thin wrapper: run token IDs through the embedding layer.

    Example:
        bx shape (4, 8)  →  embeddings shape (4, 8, 128)
    """
    return embedding_layer(token_ids)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo() -> None:
    """
    Walk through Phase 2: token IDs → embedding vectors.

    Run: python -m src.mini_gpt.phase02_embeddings
    """
    print("=" * 60)
    print("PHASE 2 DEMO: Token Embeddings")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "sample.txt"
    C = DEFAULT_EMBED_DIM  # 128
    T = 8

    # --- Step 1: Get token IDs from Phase 1 pipeline ---
    print("\n--- Step 1: Load token IDs (Phase 1) ---")
    token_ids = load_and_tokenize(data_path)
    x, y = build_input_target_pairs(token_ids, block_size=T)
    dataset = TextTokenDataset(x, y)
    loader = create_dataloader(dataset, batch_size=4, shuffle=False)
    bx, by = next(iter(loader))
    print(f"bx.shape = {tuple(bx.shape)}  (B, T) — still integers!")

    # --- Step 2: Create embedding layer ---
    print("\n--- Step 2: Create TokenEmbedding layer ---")
    V = get_vocab_size("gpt2")
    embed_layer = TokenEmbedding(vocab_size=V, embed_dim=C)
    print(f"V={V}, C={C}, weight.shape = {tuple(embed_layer.weight.shape)}  (V, C)")

    # --- Step 3: Forward pass — integers → floats ---
    print("\n--- Step 3: Forward pass (lookup) ---")
    print(f"token_ids.shape = {tuple(bx.shape)}  (B, T)")
    embeddings = embed_layer(bx)
    print(f"embeddings.shape = {tuple(embeddings.shape)}  (B, T, C)")

    print(f"\nBEFORE embedding: bx.dtype = {bx.dtype}  (integers)")
    print(f"AFTER  embedding: embeddings.dtype = {embeddings.dtype}  (floats)")

    # --- Step 4: Inspect one token ---
    print("\n--- Step 4: One token deep-dive ---")
    tokenizer = tiktoken.get_encoding("gpt2")
    token_id = bx[0, 0].item()
    token_text = tokenizer.decode([token_id])
    vector = embed_layer.get_row(token_id)

    print(f"bx[0, 0] = token_id {token_id}  →  decoded: {token_text!r}")
    print(f"embedding vector shape = {tuple(vector.shape)}  (C,) where C={C}")
    print(f"first 5 values of vector: {vector[:5].tolist()}")

    # --- Step 5: Show what rows/columns mean ---
    print("\n--- Step 5: Matrix meaning ---")
    print("W_E weight matrix (V, C):")
    print(f"  Rows    = {V} vocabulary tokens (row k = embedding for token ID k)")
    print(f"  Columns = {C} learned features per token")
    print(f"embeddings (B, T, C):")
    print(f"  Dim 0 (B={bx.shape[0]}) = batch examples")
    print(f"  Dim 1 (T={T}) = token positions in sequence")
    print(f"  Dim 2 (C={C}) = feature vector at each position")

    # --- Step 6: Same token ID → same vector everywhere ---
    print("\n--- Step 6: Same ID → same row (consistency check) ---")
    # Find another occurrence of same token_id in batch if possible
    same_mask = bx == token_id
    if same_mask.any():
        emb_at_first = embeddings[0, 0]
        other_pos = same_mask.nonzero(as_tuple=False)[0]
        emb_at_other = embeddings[other_pos[0], other_pos[1]]
        assert torch.allclose(emb_at_first, emb_at_other)
        print(f"Token {token_id} has identical embedding wherever it appears. OK.")

    print("\n" + "=" * 60)
    print("Phase 2 complete. See docs/phase02/theory.md")
    print("Next: Phase 3 (Positional Embeddings) — waiting for your go-ahead.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
