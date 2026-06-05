# Mini GPT — From First Principles

An educational project to build a decoder-only GPT-style language model from scratch using PyTorch.

**Goal:** Understand every tensor, every layer, and every equation — not just call a library.

## Project Structure

```
SLM/
├── docs/
│   ├── Prereq/           # Prerequisites (tensors)
│   ├── phase01/          # Dataset creation
│   ├── phase02/          # Token embeddings
│   └── phase03/          # Positional embeddings (next)
├── src/mini_gpt/         # PyTorch implementations (one module per phase)
├── tests/                # Unit tests for each phase
├── data/                 # Small training corpora
└── SHAPES.md             # Living reference of every tensor shape
```

## Phases

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Dataset Creation | ✅ |
| 2 | Token Embeddings | ✅ |
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

## Docs

See [`docs/README.md`](docs/README.md) for the full index.

| Phase | Theory | Code walkthrough |
|-------|--------|------------------|
| 1 | `docs/phase01/theory.md` | `docs/phase01/code_walkthrough.md` |
| 2 | `docs/phase02/theory.md` | `docs/phase02/code_walkthrough.md` |
| Prereq | `docs/Prereq/phase00_Tensor.md` | — |
| Shapes | `SHAPES.md` | — |

## Running

**Phase 1:**

```bash
source .venv/bin/activate
python -m src.mini_gpt.phase01_dataset
pytest tests/test_phase01_dataset.py -v
```

**Phase 2:**

```bash
source .venv/bin/activate
python -m src.mini_gpt.phase02_embeddings
pytest tests/test_phase02_embeddings.py -v
```

**All tests:**

```bash
pytest tests/ -v
```

## Rules

- No Hugging Face transformer implementations
- Tokenizer (tiktoken) is allowed; all transformer components are manual
- Every tensor shape is documented in `SHAPES.md`
