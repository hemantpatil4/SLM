# Phase 2: Token Embeddings

> **Goal:** Convert integer token IDs `(B, T)` into continuous vectors `(B, T, C)` that a neural network can learn from.

**Code:** `src/mini_gpt/phase02_embeddings.py`  
**Prerequisites:** [Phase 1 theory](../phase01/theory.md)  
**Code walkthrough:** [code_walkthrough.md](code_walkthrough.md)

---

## 0. Why Do We Need Embeddings?

Phase 1 gave us **integers**:

```
bx = [[40, 1842, 9552, 290, ...],   ...]   shape (B, T)
```

Neural networks cannot learn much from raw integers like `40` vs `9552` — the number `9552` is not "bigger in meaning" than `40`. They are just **labels** (names), like jersey numbers on players.

**Embeddings** convert each label into a **list of real numbers** (a vector) where:

- similar meanings → similar vectors (learned during training)
- the model can do math on these vectors (add, multiply, attention — later phases)

```
BEFORE:  token_id = 40        (one integer)
AFTER:   vector  = [0.02, -0.11, 0.05, ..., 0.08]   (C floats)
```

---

## 1. The Embedding Matrix `W_E`

The core object is a **lookup table** stored as a matrix:

```
W_E  shape (V, C)

V = vocabulary size  (≈ 50,257 for GPT-2)
C = embedding dimension (128 in our Mini GPT)
```

### What rows represent

**Each ROW is one vocabulary token.**

```
Row 0    → embedding for token ID 0
Row 40   → embedding for token ID 40  ("I")
Row 1842 → embedding for token ID 1842 (" love")
...
Row 50256 → embedding for last token
```

### What columns represent

**Each COLUMN is one feature dimension.**

```
Column 0 → feature #0  (learned — no human name, model discovers meaning)
Column 1 → feature #1
...
Column 127 → feature #127   (when C=128)
```

Together, one row is a **C-dimensional vector** describing that token.

---

## 2. Visual: The Lookup Table

Think of `W_E` as a **phone book**:

```
         feat0  feat1  feat2  ...  feat127
         (col0) (col1) (col2)      (col127)
token 0 [ 0.01   0.03  -0.02  ...   0.01  ]  ← row 0
token 1 [ 0.02  -0.01   0.04  ...  -0.03  ]  ← row 1
...
token 40[ 0.05  -0.11   0.02  ...   0.08  ]  ← row 40 = "I"
...
```

**Lookup** = open the book at row `k` and read that entire row.

---

## 3. Mathematics

For token ID `k` at batch position `(b, t)`:

```
output[b, t, :] = W_E[k, :]
```

In words: **copy row `k` of the embedding matrix.**

This is **not** a dot product with one-hot encoding in practice (though mathematically equivalent):

```
one_hot(k) @ W_E  =  row k of W_E
```

PyTorch does the fast version: **indexing**.

```python
embeddings = self.weight[token_ids]   # (B,T) indexes (V,C) → (B,T,C)
```

---

## 4. Tensor Shape Journey

| Step | Tensor | Shape | Dtype |
|------|--------|-------|-------|
| Phase 1 output | `bx` | `(B, T)` | `long` (integers) |
| Embedding matrix | `W_E` / `weight` | `(V, C)` | `float32` |
| Phase 2 output | `embeddings` | `(B, T, C)` | `float32` |

### Reading `(B, T, C)` — 3D tensor

```
embeddings[b, t, c]

Dim 0 (b) = which example in the batch
Dim 1 (t) = which token position in the sequence
Dim 2 (c) = which feature in the embedding vector
```

**Analogy:** A stack of `B` worksheets. Each worksheet has `T` rows (one per token) and `C` columns (features).

---

## 5. Numerical Example (Tiny)

Suppose `V=5`, `C=4`, `B=1`, `T=3`:

**Embedding matrix `W_E` (5 rows, 4 columns):**

```
        c0    c1    c2    c3
id 0 [ 0.1,  0.2,  0.3,  0.4 ]
id 1 [ 0.5,  0.6,  0.7,  0.8 ]
id 2 [ 0.9,  1.0,  1.1,  1.2 ]
id 3 [ 1.3,  1.4,  1.5,  1.6 ]
id 4 [ 1.7,  1.8,  1.9,  2.0 ]
```

**Input token IDs:**

```
token_ids = [[1, 4, 2]]   shape (1, 3)
```

**Lookup result:**

```
embeddings[0, 0, :] = row 1 = [0.5, 0.6, 0.7, 0.8]
embeddings[0, 1, :] = row 4 = [1.7, 1.8, 1.9, 2.0]
embeddings[0, 2, :] = row 2 = [0.9, 1.0, 1.1, 1.2]

embeddings shape (1, 3, 4)
```

---

## 6. Why This Layer Exists

| Without embeddings | With embeddings |
|--------------------|-----------------|
| Token 40 and 41 are "close numerically" | Vectors are learned — meaning drives similarity |
| Can't apply linear layers or attention | Continuous vectors work with matrix math |
| No notion of "word similarity" | Training pulls similar contexts together |

Every transformer model has this as the **first** layer after tokenization.

---

## 7. Learnable Parameters

`W_E` is **learned during training** (Phase 12):

- Initialized with small random numbers (`randn * 0.02`)
- Updated by backpropagation
- `requires_grad = True` on the weight

Total embedding parameters:

```
V × C = 50257 × 128 ≈ 6.4 million parameters
```

Just for this one matrix!

---

## 8. Important Properties

### Same token ID → same vector everywhere

If token `40` ("I") appears at position 0 and position 5, both get **identical** row 40 from `W_E`.

Position-specific meaning comes in **Phase 3** (positional embeddings).

### Integers in, floats out

```
bx.dtype          = torch.int64
embeddings.dtype  = torch.float32
```

### Token IDs must be valid

Every ID must be in `[0, V-1]`. Invalid IDs cause index errors.

---

## 9. Mini GPT Hyperparameters (This Phase)

| Symbol | Value | Meaning |
|--------|-------|---------|
| `V` | 50,257 | GPT-2 vocabulary |
| `C` | 128 | embedding dimension |
| `W_E` | (50257, 128) | learned matrix |

---

## 10. Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Float token IDs | Indexing fails or wrong lookup | `dtype=torch.long` |
| Wrong shape `(T,)` instead of `(B,T)` | Broadcasting surprises | Keep batch dimension |
| Thinking columns = tokens | Rows = tokens, columns = features | Re-read Section 1 |
| Expecting different vectors for same ID | Same ID = same row always | Add position info in Phase 3 |
| Huge V on CPU for tests | Slow / memory | Use small V=100 in unit tests |

---

## 11. Debugging Tips

```python
print(embeddings.shape)           # (B, T, C)
print(layer.weight.shape)         # (V, C)
print(embeddings[0, 0, :5])       # first 5 features at batch 0, pos 0
print(layer.get_row(40)[:5])      # embedding for token "I"
```

---

## 12. What Comes Next (Phase 3)

Token embeddings tell the model **what** token is at each position, but not **where** it is. `"I"` at position 0 and position 5 get the **same** vector.

Phase 3 adds **positional embeddings** so the model knows order:

```
final = token_embedding + position_embedding   → still (B, T, C)
```

See [`../phase03/README.md`](../phase03/README.md).

---

## 13. Run & Verify

```bash
source .venv/bin/activate
pytest tests/test_phase02_embeddings.py -v
python -m src.mini_gpt.phase02_embeddings
```
