"""
Unit tests for Phase 2: Token Embeddings.

Run: pytest tests/test_phase02_embeddings.py -v
"""

import torch
import torch.nn as nn

from src.mini_gpt.phase02_embeddings import (
    DEFAULT_EMBED_DIM,
    TokenEmbedding,
    embed_batch,
    get_vocab_size,
)


class TestVocabSize:
    def test_gpt2_vocab_size(self):
        V = get_vocab_size("gpt2")
        assert V == 50257


class TestTokenEmbeddingInit:
    def test_weight_shape(self):
        V, C = 100, 16
        layer = TokenEmbedding(vocab_size=V, embed_dim=C)
        assert layer.weight.shape == (V, C)
        assert isinstance(layer.weight, nn.Parameter)

    def test_weight_is_trainable(self):
        layer = TokenEmbedding(vocab_size=50, embed_dim=8)
        assert layer.weight.requires_grad is True


class TestTokenEmbeddingForward:
    def test_output_shape(self):
        V, C, B, T = 100, 16, 3, 10
        layer = TokenEmbedding(vocab_size=V, embed_dim=C)
        token_ids = torch.randint(0, V, (B, T), dtype=torch.long)
        out = layer(token_ids)
        assert out.shape == (B, T, C)
        assert out.dtype == torch.float32

    def test_lookup_matches_manual_row_fetch(self):
        """embedding[b,t] must equal weight[token_ids[b,t]]."""
        V, C = 50, 8
        layer = TokenEmbedding(vocab_size=V, embed_dim=C)
        token_ids = torch.tensor([[1, 5, 3], [9, 0, 7]], dtype=torch.long)

        out = layer(token_ids)

        for b in range(2):
            for t in range(3):
                tid = token_ids[b, t].item()
                assert torch.allclose(out[b, t], layer.weight[tid])

    def test_same_id_same_vector(self):
        V, C = 30, 4
        layer = TokenEmbedding(vocab_size=V, embed_dim=C)
        token_ids = torch.tensor([[7, 7, 7]], dtype=torch.long)  # same ID repeated
        out = layer(token_ids)
        assert torch.allclose(out[0, 0], out[0, 1])
        assert torch.allclose(out[0, 1], out[0, 2])

    def test_requires_long_dtype(self):
        layer = TokenEmbedding(vocab_size=10, embed_dim=4)
        float_ids = torch.randn(2, 3)  # wrong dtype
        try:
            layer(float_ids)
            raised = False
        except AssertionError:
            raised = True
        assert raised

    def test_default_embed_dim_constant(self):
        assert DEFAULT_EMBED_DIM == 128


class TestEmbedBatch:
    def test_wrapper(self):
        layer = TokenEmbedding(vocab_size=20, embed_dim=6)
        ids = torch.randint(0, 20, (2, 4), dtype=torch.long)
        out = embed_batch(ids, layer)
        assert out.shape == (2, 4, 6)


class TestGradients:
    def test_backward_updates_weight(self):
        """Embedding weights should receive gradients from a loss."""
        V, C = 20, 4
        layer = TokenEmbedding(vocab_size=V, embed_dim=C)
        ids = torch.randint(0, V, (1, 3), dtype=torch.long)
        out = layer(ids)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == (V, C)
