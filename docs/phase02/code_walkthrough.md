# Phase 2: Code Walkthrough

> **Read this file** to understand `src/mini_gpt/phase02_embeddings.py` line by line.
>
> Theory: [`theory.md`](theory.md)  
> Phase 1 pipeline: [`../phase01/code_walkthrough.md`](../phase01/code_walkthrough.md)

---

## Table of Contents

1. [Big Picture](#1-big-picture)
2. [New PyTorch Concepts](#2-new-pytorch-concepts)
3. [Imports & Connection to Phase 1](#3-imports--connection-to-phase-1)
4. [Function & Class Breakdown](#4-function--class-breakdown)
5. [Before vs After Embedding](#5-before-vs-after-embedding)
6. [The `demo()` Pipeline](#6-the-demo-pipeline)
7. [Cheat Sheet](#7-cheat-sheet)

---

## 1. Big Picture

```
Phase 1: bx (B, T) integers
              ↓
Phase 2: TokenEmbedding.forward()
              ↓
         embeddings (B, T, C) floats
              ↓
Phase 3+: positional encoding, attention, ...
```

Phase 2 adds **one new dimension** `C` and changes **dtype** from integer to float.

---

## 2. New PyTorch Concepts

### 2.1 `torch.nn.Module` — Base Class for Neural Network Layers

```python
class TokenEmbedding(nn.Module):
```

Any layer you want to train inherits from `nn.Module`. It gives you:

- automatic parameter tracking
- `.to(device)` for GPU
- integration with optimizers

---

### 2.2 `nn.Parameter` — A Tensor That Gets Trained

```python
self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim) * 0.02)
```

| Part | Meaning |
|------|---------|
| `torch.randn(V, C)` | random matrix shape (V, C) |
| `* 0.02` | scale down (small init) |
| `nn.Parameter(...)` | register as **learnable weight** |

When you call `loss.backward()`, gradients flow into `self.weight.grad`.

---

### 2.3 `super().__init__()` — Initialize Parent Class

```python
def __init__(self, vocab_size, embed_dim):
    super().__init__()   # MUST call this first for nn.Module
```

Without it, PyTorch won't track your parameters correctly.

---

### 2.4 Advanced Indexing — The Lookup

```python
embeddings = self.weight[token_ids]
```

| Tensor | Shape | Content |
|--------|-------|---------|
| `self.weight` | `(V, C)` | full embedding table |
| `token_ids` | `(B, T)` | integer indices |
| `embeddings` | `(B, T, C)` | looked-up rows |

**How it works:** for each `token_ids[b, t]`, PyTorch copies `weight[token_ids[b, t], :]` into `embeddings[b, t, :]`.

---

### 2.5 `forward()` — What Happens on a Forward Pass

```python
embeddings = embed_layer(bx)   # calls embed_layer.forward(bx)
```

Calling the module like a function runs `forward()`.

---

## 3. Imports & Connection to Phase 1

```python
from src.mini_gpt.phase01_dataset import (
    build_input_target_pairs,
    create_dataloader,
    load_and_tokenize,
    TextTokenDataset,
)
```

Phase 2 **reuses** Phase 1 to get `bx` — it does not rebuild the dataset. This shows how phases stack:

```
phase01_dataset.py  →  token IDs (B, T)
phase02_embeddings.py  →  vectors (B, T, C)
```

---

## 4. Function & Class Breakdown

### 4.1 `get_vocab_size(encoding_name="gpt2")`

```python
enc = tiktoken.get_encoding(encoding_name)
return enc.n_vocab   # 50257
```

Returns `V` — how many rows the embedding matrix needs.

---

### 4.2 `TokenEmbedding.__init__(vocab_size, embed_dim)`

**Runs once** when you create the layer.

```python
embed_layer = TokenEmbedding(vocab_size=50257, embed_dim=128)
```

Creates `self.weight` with shape `(50257, 128)`.

**Memory intuition:** 50257 × 128 × 4 bytes ≈ 25 MB for this matrix alone.

---

### 4.3 `TokenEmbedding.forward(token_ids)`

**Input:** `token_ids` shape `(B, T)`, dtype `long`

**Output:** `embeddings` shape `(B, T, C)`, dtype `float32`

**Assertions (learning guards):**

```python
assert token_ids.dtype == torch.long
assert token_ids.dim() == 2
```

Crash early with a clear message instead of cryptic PyTorch errors.

---

### 4.4 `get_row(token_id)`

```python
vector = embed_layer.get_row(40)   # shape (C,)
```

Direct access to one row — great for debugging "what does token 40 look like?"

---

### 4.5 `embed_batch(token_ids, embedding_layer)`

Thin wrapper — same as calling `embedding_layer(token_ids)`. Exists for readable pipeline code in demos and later training scripts.

---

## 5. Before vs After Embedding

### BEFORE — `bx` from Phase 1

```
bx  shape (4, 8)   dtype=int64

        pos0  pos1  pos2  pos3  pos4  pos5  pos6  pos7
row 0 [  40   1842  9552   290  2769  4673    13   198 ]
row 1 [  ...   ...   ...   ...   ...   ...   ...   ... ]
...

Each cell = ONE integer (token ID)
```

---

### AFTER — `embeddings` from Phase 2

```
embeddings  shape (4, 8, 128)   dtype=float32

For row 0, pos 0 (token_id=40):
embeddings[0, 0, :] = [0.023, -0.011, 0.008, ..., 0.019]   ← 128 floats

For row 0, pos 1 (token_id=1842):
embeddings[0, 1, :] = [different 128 floats from row 1842 of W_E]
```

**Visual — one position expands:**

```
BEFORE:  bx[0,0] = 40                    (scalar integer)

AFTER:   embeddings[0,0,:] = [128 floats from row 40 of W_E]
```

**Visual — full batch:**

```
bx:          (B, T)        2D table of integers
embeddings:  (B, T, C)     3D block — each cell "expands" into C numbers
```

---

## 6. The `demo()` Pipeline

| Step | What happens | Shape |
|------|--------------|-------|
| 1 | Phase 1 loads data | `bx: (4, 8)` |
| 2 | Create `TokenEmbedding(V=50257, C=128)` | `weight: (50257, 128)` |
| 3 | `embed_layer(bx)` | `embeddings: (4, 8, 128)` |
| 4 | Inspect token 40 ("I") | vector `(128,)` |
| 5 | Verify same ID → same vector | consistency check |

Run:

```bash
source .venv/bin/activate
python -m src.mini_gpt.phase02_embeddings
```

---

## 7. Cheat Sheet

| Code | Shape in | Shape out |
|------|----------|-----------|
| `TokenEmbedding(V, C)` | — | `weight (V, C)` |
| `layer(token_ids)` | `(B, T)` long | `(B, T, C)` float |
| `layer.get_row(k)` | int `k` | `(C,)` |
| `get_vocab_size()` | — | int `V` |

---

## Common Code Questions

### Why `nn.Parameter` instead of `nn.Embedding`?

Both work. We use `nn.Parameter` + indexing so you **see** the lookup explicitly:

```python
self.weight[token_ids]   # "copy these rows"
```

`nn.Embedding` does the same thing internally.

### Why `randn * 0.02`?

Weights start small so early training doesn't explode. GPT-style standard init.

### Does `y` (targets) get embedded too?

In full training (Phase 12), only **input** `x` is embedded. Targets `y` stay as integers for cross-entropy loss against vocabulary logits. We'll see this in Phase 12.

---

## Next Step

Read [`theory.md`](theory.md) Section 4–5 for the matrix pictures, then run the demo.

Say **"go"** for Phase 3 (Positional Embeddings).
