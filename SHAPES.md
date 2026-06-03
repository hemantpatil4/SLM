# Tensor Shape Reference

Living document — updated as each phase is implemented.

> **New to this?** Read **Section 0** in `docs/phase01_dataset_creation.md` first.
> It explains tensors, rows, columns, BPE, and `(B, T)` in plain language.

**Notation:**
- `B` = batch size (how many text chunks we process together = **number of rows**)
- `T` = sequence length (how many tokens per chunk = **number of columns**)
- `V` = vocabulary size (how many different tokens the tokenizer knows)

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

## Phase 2+: (to be added)
