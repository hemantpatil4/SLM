# Tensor Shape Reference

Living document — updated as each phase is implemented.

> **New to this?** Read **Section 0** in [`docs/phase01/theory.md`](docs/phase01/theory.md) first.

**Notation:**
- `B` = batch size (how many text chunks we process together = **number of rows**)
- `T` = sequence length (how many tokens per chunk = **number of columns**)
- `V` = vocabulary size (how many different tokens the tokenizer knows)
- `C` = embedding dimension (how many features per token vector)

**Remember:** A 2D tensor is like a spreadsheet. **Rows** = examples. **Columns** = token positions.

---

## Phase 1: Dataset Creation

| Tensor / Object | Shape | Rows | Columns |
|-----------------|-------|------|---------|
| Raw text | string | — | — |
| Full token sequence | `(N,)` | N = total tokens in corpus | each element is one token ID (scalar) |
| Input batch (`x`) | `(B, T)` | B sequences | T token positions (input tokens) |
| Target batch (`y`) | `(B, T)` | B sequences | T token positions (next-token targets) |
| Single input row | `(T,)` | — | T consecutive input token IDs |
| Single target row | `(T,)` | — | T consecutive target token IDs (shifted by +1) |

### Input–Target Shift Relationship

For one sequence of token IDs `[t₀, t₁, t₂, …, t_{T-1}, t_T]`:

```
Input  x = [t₀, t₁, t₂, …, t_{T-1}]   shape (T,)
Target y = [t₁, t₂, t₃, …, t_T    ]   shape (T,)
```

At position `i`: the model sees token `x[i]` and must predict token `y[i] = x[i+1]`.

---

## Phase 2: Token Embeddings

| Tensor / Object | Shape | Dim 0 | Dim 1 | Dim 2 |
|-----------------|-------|-------|-------|-------|
| Input token IDs (`token_ids`) | `(B, T)` | batch example | token position | — |
| Embedding matrix `W_E` (`weight`) | `(V, C)` | vocabulary token ID | feature dimension | — |
| Token embeddings (output) | `(B, T, C)` | batch example | token position | feature |
| Single token vector | `(C,)` | — | — | C features |
| Row `k` of `W_E` | `(C,)` | — | — | embedding for token ID `k` |

### Lookup Rule

```
token_ids[b, t] = k   (integer index)
embeddings[b, t, :] = W_E[k, :]   (copy row k)
```

### Shape Journey (Phase 1 → Phase 2)

```
bx          (B, T)      integers
   ↓ TokenEmbedding
embeddings  (B, T, C)   floats
```

Default Mini GPT: `V=50257`, `C=128`.

---

## Phase 3+: (to be added)
