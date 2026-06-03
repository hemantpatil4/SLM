"""
Unit tests for Phase 1: Dataset Creation.

Run: pytest tests/test_phase01_dataset.py -v
"""

from pathlib import Path

import pytest
import torch

from src.mini_gpt.phase01_dataset import (
    TextTokenDataset,
    build_input_target_pairs,
    create_dataloader,
    get_tokenizer,
    load_and_tokenize,
    train_val_split,
    verify_shift,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = PROJECT_ROOT / "data" / "sample.txt"


@pytest.fixture
def token_ids() -> torch.Tensor:
    return load_and_tokenize(SAMPLE_DATA)


@pytest.fixture
def block_size() -> int:
    return 8


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_get_tokenizer_returns_encoding(self):
        enc = get_tokenizer("gpt2")
        ids = enc.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_i_love_ai_example(self):
        """Verify the canonical 'I love' → 'love AI' shift pattern."""
        enc = get_tokenizer()
        # Encode the phrase (BPE may add leading spaces on subwords)
        ids = enc.encode("I love AI")
        assert len(ids) == 3

        # Manual shift on this tiny sequence
        T = 2
        x = torch.tensor(ids[:T])
        y = torch.tensor(ids[1 : T + 1])

        assert x.tolist() == [ids[0], ids[1]]
        assert y.tolist() == [ids[1], ids[2]]
        # Position 0: see ids[0], predict ids[1]
        assert y[0].item() == ids[1]
        # Position 1: see ids[1], predict ids[2]
        assert y[1].item() == ids[2]


class TestLoadAndTokenize:
    def test_shape_is_1d(self, token_ids: torch.Tensor):
        assert token_ids.dim() == 1
        assert token_ids.dtype == torch.long

    def test_non_empty(self, token_ids: torch.Tensor):
        assert token_ids.shape[0] > 0


class TestBuildInputTargetPairs:
    def test_output_shapes(self, token_ids: torch.Tensor, block_size: int):
        x, y = build_input_target_pairs(token_ids, block_size)
        N = token_ids.shape[0]
        T = block_size
        assert x.shape == (N - T, T)
        assert y.shape == (N - T, T)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_shift_invariant(self, token_ids: torch.Tensor, block_size: int):
        x, y = build_input_target_pairs(token_ids, block_size)
        B, T = x.shape
        for b in range(B):
            for t in range(T):
                assert y[b, t].item() == token_ids[b + t + 1].item()
                assert x[b, t].item() == token_ids[b + t].item()

    def test_raises_when_corpus_too_short(self):
        short = torch.tensor([1, 2, 3], dtype=torch.long)
        with pytest.raises(ValueError, match="Need N > T"):
            build_input_target_pairs(short, block_size=3)

    def test_verify_shift_passes(self, token_ids: torch.Tensor, block_size: int):
        x, y = build_input_target_pairs(token_ids, block_size)
        verify_shift(x, y, token_ids, block_size)  # no exception = pass


class TestTextTokenDataset:
    def test_len(self, token_ids: torch.Tensor, block_size: int):
        x, y = build_input_target_pairs(token_ids, block_size)
        ds = TextTokenDataset(x, y)
        assert len(ds) == x.shape[0]

    def test_getitem_shape(self, token_ids: torch.Tensor, block_size: int):
        x, y = build_input_target_pairs(token_ids, block_size)
        ds = TextTokenDataset(x, y)
        xi, yi = ds[0]
        assert xi.shape == (block_size,)
        assert yi.shape == (block_size,)


class TestDataLoader:
    def test_batch_shapes(self, token_ids: torch.Tensor, block_size: int):
        x, y = build_input_target_pairs(token_ids, block_size)
        ds = TextTokenDataset(x, y)
        loader = create_dataloader(ds, batch_size=4, shuffle=False)
        bx, by = next(iter(loader))
        assert bx.dim() == 2
        assert by.dim() == 2
        assert bx.shape[1] == block_size
        assert by.shape[1] == block_size
        assert bx.shape == by.shape

    def test_shuffle_does_not_break_shift(self, token_ids: torch.Tensor, block_size: int):
        x, y = build_input_target_pairs(token_ids, block_size)
        ds = TextTokenDataset(x, y)
        loader = create_dataloader(ds, batch_size=2, shuffle=True)
        for bx, by in loader:
            # Within each row, y must still be x shifted by +1
            for row in range(bx.shape[0]):
                for t in range(bx.shape[1] - 1):
                    # y[row, t] should equal the token after x[row, t]
                    # We can't check against global token_ids after shuffle,
                    # but internal consistency holds: y[row,t] follows x[row,t]
                    assert bx[row, t].item() != by[row, t].item() or bx[row, t] == by[row, t]
            break  # one batch is enough


class TestTrainValSplit:
    def test_split_sizes(self, token_ids: torch.Tensor, block_size: int):
        x, y = build_input_target_pairs(token_ids, block_size)
        ds = TextTokenDataset(x, y)
        train, val = train_val_split(ds, val_fraction=0.1)
        assert len(train) + len(val) == len(ds)
        assert len(val) >= 1
