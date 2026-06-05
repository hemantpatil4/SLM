# Phase 1: Dataset Creation

> **Goal:** Turn raw text into `(input, target)` tensor pairs that teach the model next-token prediction.

---

## 0. Absolute Beginner Guide (Start Here)

Read this section first if you are new to deep learning. No prior knowledge assumed.

---

### 0.1 Computers Don't Understand Words

When you type `"I love AI"`, you see three words. A computer sees **characters**: `I`, space, `l`, `o`, `v`, `e`, space, `A`, `I`.

Before a neural network can learn from text, we must:

1. **Break text into small pieces** (tokens)
2. **Replace each piece with a number** (token ID)
3. **Put those numbers in a grid** (tensor) so PyTorch can process them

Think of it like translating English into a secret number code the machine understands.

---

### 0.2 What Is a Token?

A **token** is one small piece of text. It can be:

- a whole word: `"hello"`
- part of a word: `"ing"` (from `"learning"`)
- a single character: `"a"`
- punctuation or space: `"."` or `" "`

Example:

```
Sentence:  "I love AI"
Tokens:    ["I", " love", " AI"]
            ↑      ↑       ↑
         token0  token1  token2
```

Each token gets a number called a **token ID**:

```
"I"     → 40
" love" → 1842
" AI"   → 9552
```

So `"I love AI"` becomes: `[40, 1842, 9552]`

That's just a list of numbers. The model works with numbers, not letters.

---

### 0.3 What Is BPE? (Byte Pair Encoding)

**BPE** is the method GPT uses to decide how to split text into tokens.

**Simple idea:** Start with single characters, then repeatedly merge the most common pairs into new tokens.

**Analogy — LEGO blocks:**

Imagine you only have single-letter blocks: `l`, `e`, `a`, `r`, `n`…

You notice `"le"` appears in many words (`learn`, `love`, `letter`). So you glue `l` + `e` into one block `"le"`.

Then `"ar"` appears a lot → merge into `"ar"`.

Over time you build bigger blocks: `"learn"`, `"ing"`, `" the"`, etc.

**Why BPE?**

| Approach | Problem |
|----------|---------|
| One token per **character** | Sequences get very long; hard to learn |
| One token per **whole word** | Unknown words break the system (`"ChatGPT"` never seen → ???) |
| **BPE (in between)** | Common pieces become tokens; rare words split into known pieces |

Example with GPT-2 BPE:

```
"learning"  →  ["learn", "ing"]     (2 tokens)
"ChatGPT"   →  ["Chat", "G", "PT"] (3 tokens — never seen as one word, but pieces are known)
```

**Key point:** BPE is a **vocabulary builder**. It creates a fixed list of ~50,000 token pieces. Every piece has a number (ID). We use the library `tiktoken`, which already has this list built for GPT-2.

---

### 0.4 What Is a Vocabulary?

The **vocabulary** is the complete dictionary of all tokens the model knows.

```
Vocabulary size V ≈ 50,257  (for GPT-2)

Token ID 0    → some piece
Token ID 1    → some piece
...
Token ID 40   → "I"
Token ID 1842 → " love"
...
Token ID 50256 → last token
```

- Each token ID is like a **row number** in this dictionary.
- The model will later have one learned vector per vocabulary entry (Phase 2).

---

### 0.5 What Is a Matrix? (Rows and Columns)

Before tensors, understand a **matrix** — a table of numbers, like a spreadsheet.

```
        Column 0   Column 1   Column 2
        (pos 0)    (pos 1)    (pos 2)
Row 0 [   40        1842       9552   ]   ← first training example
Row 1 [  41762       364       3672   ]   ← second training example
Row 2 [  290        2769       4673   ]   ← third training example
```

| Term | Meaning |
|------|---------|
| **Row** | One horizontal line = one training example (one chunk of text) |
| **Column** | One vertical line = one position in the sequence (1st token, 2nd token, …) |
| **Cell** | One number at `[row, column]`, e.g. `x[1, 2]` = row 1, column 2 |

Reading `x[1, 2] = 3672` means: "In the 2nd training example, at the 3rd token position, the token ID is 3672."

---

### 0.6 What Is a Tensor?

A **tensor** is what PyTorch calls a multi-dimensional array of numbers.

```
0D tensor  = one number          →  5           (shape: ())
1D tensor  = a list / vector     →  [40, 1842, 9552]   (shape: (3,))
2D tensor  = a matrix / table    →  see grid above     (shape: (3, 3))
3D tensor  = stack of matrices   →  (Phase 2+)         (shape: (B, T, C))
```

**Tensor = matrix when it has 2 dimensions.** We say "tensor" because later we'll use 3D, 4D arrays too.

In code:

```python
import torch
x = torch.tensor([[40, 1842, 9552],
                  [41762, 364, 3672]])   # a 2D tensor (matrix)
print(x.shape)   # torch.Size([2, 3])  →  2 rows, 3 columns
```

**Shape** tells you the size of each dimension: `(rows, columns)` for 2D.

---

### 0.7 What Do B and T Mean?

We use letters as **placeholders** for sizes (like algebra):

| Symbol | Name | Meaning |
|--------|------|---------|
| **B** | Batch size | How many training examples we process **at the same time** |
| **T** | Time / sequence length | How many tokens **in each example** (context window) |
| **N** | Corpus length | Total tokens in the entire text file |
| **V** | Vocabulary size | How many different tokens exist (~50,257) |

**Batch analogy — a classroom:**

- **One row** = one student's homework (one sentence chunk)
- **B rows** = B students turn in homework at once (processed in parallel)
- **T columns** = each homework has T answers in a row (T token positions)

---

### 0.8 What Does `x` Shape `(B, T)` Look Like?

**x** = **input** tokens (what the model **reads**)

Suppose `B = 3` (three examples) and `T = 4` (four tokens each):

```
x  shape (3, 4):

              col0   col1   col2   col3
              (t=0)  (t=1)  (t=2)  (t=3)
row 0 (b=0) [  40    1842   9552    290  ]   "I love AI and"
row 1 (b=1) [ 1842   9552    290   2769  ]   "love AI and deep"
row 2 (b=2) [ 9552    290   2769   4673  ]   "AI and deep learning"
```

- **Each row** = one sliding window over the text (one training example)
- **Each column** = one time step (token position 0, 1, 2, 3)
- **Each number** = a token ID (index into vocabulary)

---

### 0.9 What Does `y` Shape `(B, T)` Look Like? (The Shift)

**y** = **target** tokens (what the model must **predict**)

**y** is the same as **x**, but **shifted one step to the right**.

Same example — compare row 0:

```
Input  x[0]:  [  40,   1842,  9552,   290 ]   ← model SEES these
Target y[0]:  [ 1842,  9552,   290,  2769 ]   ← model must PREDICT these
                 ↑       ↑       ↑      ↑
              predict  predict predict predict
              after    after   after   after
               40     1842    9552     290
```

Visual shift:

```
x:  [  A  |  B  |  C  |  D  ]
y:  [  B  |  C  |  D  |  E  ]
     ↑ each target is "the next token after x"
```

Full **y** matrix (same B=3, T=4):

```
y  shape (3, 4):

              col0   col1   col2   col3
row 0 (b=0) [ 1842   9552    290   2769 ]
row 1 (b=1) [ 9552    290   2769   4673 ]
row 2 (b=2) [  290   2769   4673     13 ]
```

Notice: `y[0, 0] = 1842` is the token that comes **right after** `x[0, 0] = 40`.

**Rule in plain English:** At every cell, the target is always "whatever token comes next."

---

### 0.10 Side-by-Side: x and y as Two Grids

```
         INPUT x (what model reads)          TARGET y (what model predicts)
         ─────────────────────────          ────────────────────────────────

Position:   0      1      2      3           0      1      2      3
          ┌──────┬──────┬──────┬──────┐    ┌──────┬──────┬──────┬──────┐
Row 0     │  40  │ 1842 │ 9552 │  290 │    │ 1842 │ 9552 │  290 │ 2769 │
          └──────┴──────┴──────┴──────┘    └──────┴──────┴──────┴──────┘
               ↘      ↘      ↘      ↘
            predict predict predict predict
              each arrow: "given this x, predict this y"
```

At **column 1, row 0**:
- Model reads token `1842` (`" love"`) at `x[0, 1]`
- Correct answer is `9552` (`" AI"`) at `y[0, 1]`

That's exactly: see `"love"` → predict `"AI"`.

---

### 0.11 From One Sentence to Many Rows

We don't just use one sentence. We read a **whole file** of text, tokenize it into one long list of N tokens, then slide a window of size T:

```
Full text tokens:  [A][B][C][D][E][F][G]     N = 7 tokens

Window T = 3:

Row 0:  x = [A,B,C]  →  y = [B,C,D]
Row 1:  x = [B,C,D]  →  y = [C,D,E]
Row 2:  x = [C,D,E]  →  y = [D,E,F]
Row 3:  x = [D,E,F]  →  y = [E,F,G]

Result: x and y are both shape (4, 3)  →  B=4, T=3
```

More text = more rows = more training examples.

---

### 0.12 Glossary (Quick Reference)

| Word | Simple meaning |
|------|----------------|
| **Token** | One small piece of text |
| **Token ID** | The number assigned to that token |
| **BPE** | Method to split text into tokens (merge common pieces) |
| **Vocabulary** | Full list of all tokens the model knows |
| **Tokenizer** | Tool that converts text ↔ token IDs (`tiktoken`) |
| **Tensor** | Multi-dimensional array of numbers (PyTorch) |
| **Shape** | Sizes of each dimension, e.g. `(3, 4)` = 3 rows, 4 columns |
| **Row** | One training example (one chunk of text) |
| **Column** | One position in the sequence (time step) |
| **B** | Number of rows (batch size) |
| **T** | Number of columns (sequence length / context window) |
| **x (input)** | Tokens the model reads |
| **y (target)** | Next tokens the model must predict (x shifted by +1) |
| **Shift** | Move targets one step right so each target is the "next" token |

---

## 1. Theory — What Are We Building?

A **language model** is a function that assigns probabilities to sequences of text. GPT-style models are **decoder-only** and trained with a simple objective:

> Given tokens seen so far, predict the **next** token.

Example:

```
Text:   "I love AI"
Tokens: [I, love, AI]

At position 0: see "I"     → predict "love"
At position 1: see "love"  → predict "AI"
```

This is called **causal language modeling** or **next-token prediction**.

The model never sees the answer while making a prediction — it only sees past (and current) tokens. That constraint is enforced later with a **causal mask** in attention (Phase 4). For now, we focus on preparing the data correctly.

---

## 2. Key Concepts

### 2.1 Tokenization

Raw text is converted to **token IDs** (integers) using a tokenizer.

```
"I love"  →  tokenizer  →  [40, 1848]   (example IDs; actual values depend on vocab)
```

- Each integer is an index into a **vocabulary** of size `V`.
- We use `tiktoken` (GPT-2 BPE tokenizer) — allowed by project rules.

### 2.2 Context Window

The **context window** (also called **block size** or **sequence length**) is the maximum number of tokens the model processes at once. We denote it `T`.

```
Context window T = 4

Sequence: [t₀, t₁, t₂, t₃, t₄, t₅, t₆]
           └──── block 1 ────┘
               └──── block 2 ────┘
                   └──── block 3 ────┘
```

If the corpus has `N` tokens and context window is `T`, we can extract approximately `N - T` training examples using a **sliding window** (or `⌊N/T⌋` non-overlapping chunks — we use sliding for more data).

### 2.3 Input–Target Shifting

For autoregressive training, **input** and **target** are the same sequence shifted by one position:

```
Full token sequence:  [ 40,  1848,  9552,  13  ]
                       t₀    t₁     t₂     t₃

Input  (x):           [ 40,  1848,  9552  ]     shape (T,)  where T=3
Target (y):           [ 1848, 9552,  13  ]     shape (T,)
```

**Rule:** `y[i] = x[i+1]` for all valid `i`.

In matrix form for a batch:

```
x shape: (B, T)   — row b is one training sequence of T input tokens
y shape: (B, T)   — row b is the corresponding T target tokens (shifted)
```

| Dimension | Meaning |
|-----------|---------|
| **Row** (dim 0) | One independent training example (one chunk of text) |
| **Column** (dim 1) | One position in the sequence (time step) |

At column `i`, row `b`: model input is `x[b, i]`, correct answer is `y[b, i]`.

---

## 3. Mathematical Formulation

Given a token sequence **s** = (s₁, s₂, …, s_N):

For each valid starting index `j`:

```
x_j = (s_j, s_{j+1}, …, s_{j+T-1})      ∈ ℤ^T
y_j = (s_{j+1}, s_{j+2}, …, s_{j+T})    ∈ ℤ^T
```

Training minimizes **cross-entropy** (Phase 12):

```
L = - (1 / (B·T))  Σ_b Σ_i  log P(y_{b,i} | x_{b,0}, …, x_{b,i})
```

Intuition: at each position `i`, the model should assign high probability to the true next token `y_{b,i}`.

---

## 4. Numerical Example

**Text:** `"I love AI"`

Assume tokenizer produces:

```
"I"    → 40
" love" → 1848   (note: BPE often includes leading space)
" AI"  → 9552
```

Full sequence: `[40, 1848, 9552]` — length N = 3.

With **context window T = 2**:

| Start j | Input x | Target y |
|---------|---------|----------|
| 0 | `[40, 1848]` | `[1848, 9552]` |

Only one block fits (N - T = 1 example).

**Interpretation:**
- Position 0: input token `40` ("I") → target `1848` (" love")
- Position 1: input token `1848` (" love") → target `9552` (" AI")

This matches the project example:

```
Input:  [I, love]
Target: [love, AI]
```

With **T = 64** (our eventual default), each training row is 64 tokens wide; shorter corpora may yield fewer blocks.

---

## 5. PyTorch Implementation Overview

Components:

1. **`load_and_tokenize`** — read file, return 1D LongTensor of all token IDs
2. **`build_input_target_pairs`** — split 1D tensor into `(x, y)` with shift
3. **`TextTokenDataset`** — `torch.utils.data.Dataset` yielding `(x, y)` chunks
4. **`create_dataloader`** — batched DataLoader with shuffling

---

## 6. Debugging Tips

1. **Print shapes after every step:**
   ```python
   print(f"token_ids.shape = {token_ids.shape}")  # expect (N,)
   print(f"x.shape = {x.shape}, y.shape = {y.shape}")  # expect (B, T)
   ```

2. **Decode a row** to verify shift:
   ```python
   print(tokenizer.decode([x[0, 0].item()]))   # first input token
   print(tokenizer.decode([y[0, 0].item()]))   # should be "next" token
   ```

3. **Assert the shift invariant:**
   ```python
   assert torch.equal(y[:, :-1], x[:, 1:])  # wrong — see note below
   ```
   Actually: `y[b, i] == x[b, i+1]` only when x and y come from the same contiguous slice. For our construction: `y[b, i] == full_sequence[start + i + 1]` and `x[b, i] == full_sequence[start + i]`.

4. **Check dtype:** token IDs must be `torch.long` (int64) for `nn.Embedding`.

---

## 7. Common Mistakes

| Mistake | Why it's wrong | Fix |
|---------|----------------|-----|
| Input and target same shape but not shifted | Model learns identity | Always set `y = x shifted by +1` |
| Using float for token IDs | Embedding expects LongTensor | `.long()` or `dtype=torch.long` |
| Forgetting `<|endoftext|>` or newlines | Broken sentence boundaries | Keep natural text; tokenizer handles spaces |
| Context window > corpus length | Zero training samples | Ensure N > T or reduce T |
| Shuffling tokens within a sequence | Destroys word order | Only shuffle **rows** (batch), not columns |
| Confusing batch dim and time dim | Wrong attention later | Batch = dim 0, Time = dim 1 → `(B, T)` |

---

## 8. Visualization

Sliding window over token stream:

```
Tokens:  [A][B][C][D][E][F][G]
          └─T=3─┘
              └─T=3─┘
                  └─T=3─┘

Block 0:  x=[A,B,C]  y=[B,C,D]
Block 1:  x=[B,C,D]  y=[C,D,E]
Block 2:  x=[C,D,E]  y=[D,E,F]
Block 3:  x=[D,E,F]  y=[E,F,G]
```

Each arrow is one "predict next" step inside the block.

---

## 9. What Comes Next (Phase 2)

Token IDs `(B, T)` will be passed through an **embedding matrix** `(V, C)` to produce `(B, T, C)` continuous vectors — one C-dimensional vector per token per position.

See [`../phase02/theory.md`](../phase02/theory.md) and `SHAPES.md` for the running shape reference.

---

## 10. Code Walkthrough

For a **line-by-line explanation** of `phase01_dataset.py` (PyTorch, classes, `__init__`, batching before/after), see:

**[`code_walkthrough.md`](code_walkthrough.md)**

---

## 11. Run & Verify

On macOS, `python` may not exist until you activate the project venv:

```bash
cd /path/to/SLM
source .venv/bin/activate
pytest tests/test_phase01_dataset.py -v
python -m src.mini_gpt.phase01_dataset
```

Or without activating:

```bash
.venv/bin/python -m src.mini_gpt.phase01_dataset
```
