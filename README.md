# Mini GPT — From First Principles

An educational project to build a decoder-only GPT-style language model from scratch using PyTorch.

**Goal:** Understand every tensor, every layer, and every equation — not just call a library.

## Project Structure

```
SLM/
├── docs/                  # One markdown file per phase (theory + intuition)
├── src/mini_gpt/          # PyTorch implementations (one module per phase)
├── tests/                 # Unit tests for each phase
├── data/                  # Small training corpora
└── SHAPES.md              # Living reference of every tensor shape
```

## Phases

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Dataset Creation | ✅ |
| 2 | Token Embeddings | — |
| 3 | Positional Embeddings | — |
| 4 | Single-Head Masked Self-Attention | — |
| 5 | Multi-Head Attention | — |
| 6 | Feed-Forward Network | — |
| 7 | Residual Connections | — |
| 8 | Layer Normalization | — |
| 9 | Dropout | — |
| 10 | GPT Decoder Block | — |
| 11 | Mini GPT Model | — |
| 12 | Training | — |
| 13 | Text Generation | — |

## Setup

On macOS, use `python3` (there is often no `python` command):

```bash
cd /path/to/SLM
python3 -m venv .venv
source .venv/bin/activate    # after this, `python` works inside the venv
pip install -r requirements.txt
```

## Running Phase 1

**Option A — activate the virtual environment first (recommended):**

```bash
source .venv/bin/activate
python -m src.mini_gpt.phase01_dataset
pytest tests/test_phase01_dataset.py -v
```

**Option B — without activating (one-shot):**

```bash
.venv/bin/python -m src.mini_gpt.phase01_dataset
.venv/bin/pytest tests/test_phase01_dataset.py -v
```

## Rules

- No Hugging Face transformer implementations
- Tokenizer (tiktoken) is allowed; all transformer components are manual
- Every tensor shape is documented in `SHAPES.md`
