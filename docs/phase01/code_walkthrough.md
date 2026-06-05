# Phase 1: Code Walkthrough

> **Read this file** to understand `src/mini_gpt/phase01_dataset.py` line by line.
>
> For theory (BPE, tensors, x/y grids), see [`theory.md`](theory.md).
> For tensor basics, see [`../Prereq/phase00_Tensor.md`](../Prereq/phase00_Tensor.md).

---

## Table of Contents

1. [Big Picture — What Happens in Order](#1-big-picture--what-happens-in-order)
2. [PyTorch Concepts Used in This File](#2-pytorch-concepts-used-in-this-file)
3. [Python Concepts Used in This File](#3-python-concepts-used-in-this-file)
4. [Imports — What We Load](#4-imports--what-we-load)
5. [Function by Function](#5-function-by-function)
6. [The `TextTokenDataset` Class](#6-the-texttokendataset-class)
7. [Before Batching vs After Batching](#7-before-batching-vs-after-batching)
8. [The `demo()` Function — Full Pipeline](#8-the-demo-function--full-pipeline)
9. [Reading the Terminal Output](#9-reading-the-terminal-output)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. Big Picture — What Happens in Order

Think of the pipeline like a factory assembly line:

```
sample.txt (raw text)
    ↓  load_and_tokenize()
token_ids  shape (N,)     ← one long list of numbers
    ↓  build_input_target_pairs()
x, y       shape (num_blocks, T)   ← two tables (input + target)
    ↓  TextTokenDataset()
dataset    can fetch one row at a time
    ↓  create_dataloader()
batches    shape (B, T)     ← groups of rows stacked together
    ↓  (later phases)
model training
```

**Before batching:** data lives in big tables `x` and `y`. You can grab **one row** at a time.

**After batching:** DataLoader grabs **B rows at once** and stacks them into `bx` and `by`.

---

## 2. PyTorch Concepts Used in This File

### 2.1 `import torch`

PyTorch is the library for:
- storing numbers in **tensors** (GPU-friendly arrays)
- building neural networks (Phase 2+)
- training with automatic gradients (Phase 12)

We only use the **tensor** and **data loading** parts in Phase 1.

---

### 2.2 `torch.tensor(...)` — Create a Tensor

```python
ids = [40, 1842, 9552]
token_ids = torch.tensor(ids, dtype=torch.long)
```

| Part | Meaning |
|------|---------|
| `[40, 1842, 9552]` | plain Python list of integers |
| `torch.tensor(...)` | convert list → PyTorch tensor |
| `dtype=torch.long` | 64-bit integers (required for token IDs) |

Result:

```
token_ids = tensor([40, 1842, 9552])
token_ids.shape = (3,)     ← 1D, length 3
token_ids.dtype = torch.int64
```

---

### 2.3 `.shape` — How Big Is the Tensor?

```python
token_ids.shape   # torch.Size([47])  →  47 tokens, 1 dimension
x.shape           # torch.Size([39, 8]) →  39 rows, 8 columns
```

Always print `.shape` when debugging. It tells you if your data layout is correct.

---

### 2.4 Slicing — Cut Out Pieces

Works like Python lists, but on tensors:

```python
token_ids[0]        # first token          → scalar tensor
token_ids[0:3]      # tokens 0,1,2         → shape (3,)
token_ids[5:10]     # tokens 5..9          → shape (5,)

x[0]                # row 0 of matrix      → shape (T,)
x[0, 2]             # row 0, column 2      → one number
x[b : b + T]        # slice from index b, length T
```

**In our code:**

```python
x[b] = token_ids[b : b + T]           # input window
y[b] = token_ids[b + 1 : b + T + 1]   # same window, shifted +1
```

---

### 2.5 `torch.zeros(rows, cols)` — Empty Table Filled with 0

```python
x = torch.zeros(39, 8, dtype=torch.long)
```

Creates a **39 × 8** table of zeros. We fill each row in a loop.

Why zeros? We need a fixed-size container before we copy real token IDs in.

---

### 2.6 `.item()` — Tensor → Plain Python Number

```python
y[0, 0]        # tensor(1842)  ← still a tensor (0-dimensional)
y[0, 0].item() # 1842          ← plain Python int
```

Use `.item()` when you want one number for printing or `assert`.

---

### 2.7 `.tolist()` — Tensor → Python List

```python
x[0].tolist()   # [40, 1842, 9552, 290, 2769, 4673, 13, 198]
```

Useful for `tokenizer.decode(...)` which expects a list of ints.

---

### 2.8 `torch.utils.data.Dataset` — Blueprint for "Give Me One Example"

PyTorch defines a **protocol** (rules) for datasets:

| Method | Must return | Purpose |
|--------|---------------|---------|
| `__len__()` | integer | how many examples exist |
| `__getitem__(idx)` | one example | example at index `idx` |

If your class follows these rules, `DataLoader` knows how to use it.

---

### 2.9 `DataLoader` — Groups Examples into Batches

```python
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for bx, by in loader:
    # bx shape (4, T), by shape (4, T)
    ...
```

| Argument | Meaning |
|----------|---------|
| `batch_size=4` | grab 4 rows per batch |
| `shuffle=True` | randomize **row order** each epoch |
| `drop_last=False` | keep the last batch even if it has fewer than 4 rows |

---

### 2.10 `random_split` — Train / Validation Split

```python
train_set, val_set = random_split(dataset, [n_train, n_val], generator=...)
```

Randomly divides the dataset into two smaller subsets for training vs checking accuracy.

---

## 3. Python Concepts Used in This File

### 3.1 `def function_name(...) -> return_type:`

The `-> torch.Tensor` part is a **type hint**. It says "this function returns a tensor." Python does not enforce it — it's documentation for you and your editor.

```python
def load_and_tokenize(file_path: str | Path) -> torch.Tensor:
```

`str | Path` means: you can pass a string **or** a `Path` object.

---

### 3.2 `tuple[...]` — Function Returns Two Things

```python
def build_input_target_pairs(...) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    return x, y
```

`return x, y` in Python automatically packs into a **tuple** `(x, y)`.

Unpacking on the caller side:

```python
x, y = build_input_target_pairs(token_ids, block_size=8)
# x is first tensor, y is second tensor
```

Same pattern for `__getitem__`:

```python
def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    return self.x[idx], self.y[idx]   # returns (input_row, target_row)
```

---

### 3.3 `class TextTokenDataset(Dataset):` — Define a Class

A **class** is a blueprint for creating objects that hold data + behavior.

```python
class TextTokenDataset(Dataset):
    def __init__(self, x, y): ...   # constructor — runs when you create object
    def __len__(self): ...          # how many items
    def __getitem__(self, idx): ... # fetch one item
```

**Creating an object (instance):**

```python
dataset = TextTokenDataset(x, y)
#          ↑ calls __init__(self, x, y)
```

---

### 3.4 `__init__` — Constructor (Setup)

Runs **once** when you write `TextTokenDataset(x, y)`.

```python
def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
    assert x.shape == y.shape, "input and target must have same shape"
    self.x = x   # save x inside the object as self.x
    self.y = y   # save y inside the object as self.y
```

| Line | What it does |
|------|--------------|
| `assert x.shape == y.shape` | crash early if x and y tables don't match |
| `self.x = x` | attach tensor `x` to this object forever |
| `self.y = y` | attach tensor `y` to this object forever |

`self` means "this specific dataset object."

---

### 3.5 `__len__` — How Many Examples?

```python
def __len__(self) -> int:
    return self.x.shape[0]   # number of rows
```

Lets you write:

```python
len(dataset)   # calls dataset.__len__()  →  39
```

---

### 3.6 `__getitem__` — Fetch One Example by Index

```python
def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    return self.x[idx], self.y[idx]
```

Lets you write:

```python
xi, yi = dataset[0]   # calls dataset.__getitem__(0)
# xi shape (T,), yi shape (T,)
```

**Index `0`** = first row. **Index `5`** = sixth row.

---

### 3.7 `enumerate(loader)` — Loop with Batch Number

```python
for batch_idx, (bx, by) in enumerate(loader):
    print(batch_idx)   # 0, 1, 2, ...
```

| Part | Meaning |
|------|---------|
| `enumerate(loader)` | yields `(0, batch0)`, `(1, batch1)`, ... |
| `batch_idx` | which batch we're on |
| `(bx, by)` | unpacked tuple from each batch |

Each `batch` from DataLoader is already `(input_batch, target_batch)`.

---

### 3.8 `Path` — File Paths Made Easy

```python
from pathlib import Path

file_path = Path("data/sample.txt")
text = file_path.read_text(encoding="utf-8")
```

Cleaner than string concatenation for finding files.

---

### 3.9 `if __name__ == "__main__":`

```python
if __name__ == "__main__":
    demo()
```

Means: **only run `demo()` when you execute this file directly**, not when another file imports it.

That's why this works:

```bash
python -m src.mini_gpt.phase01_dataset
```

---

## 4. Imports — What We Load

```python
import tiktoken                              # GPT-2 tokenizer (text ↔ numbers)
import torch                                 # tensors
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path                     # file paths
```

| Import | Used for |
|--------|----------|
| `tiktoken` | `get_tokenizer()`, encode/decode text |
| `torch` | create tensors, zeros, long dtype |
| `Dataset` | parent class for `TextTokenDataset` |
| `DataLoader` | batching |
| `random_split` | train/val split |
| `Path` | read `data/sample.txt` |

---

## 5. Function by Function

### 5.1 `get_tokenizer(name="gpt2")`

**What it does:** Returns the GPT-2 BPE tokenizer object.

```python
tokenizer = get_tokenizer()
ids = tokenizer.encode("I love AI")    # text → [40, 1842, 9552]
text = tokenizer.decode([40, 1842])    # numbers → "I love"
```

**No tensors yet** — just Python lists and strings.

---

### 5.2 `load_and_tokenize(file_path)`

**What it does:** Read a `.txt` file → return one long 1D tensor of all token IDs.

**Step by step:**

```
sample.txt
"I love AI and deep learning.
 Transformers learn..."
        ↓ read_text()
one big Python string
        ↓ tokenizer.encode()
[40, 1842, 9552, 290, 2769, 4673, 13, 198, ...]   list of 47 ints
        ↓ torch.tensor(..., dtype=torch.long)
tensor([40, 1842, 9552, ...])   shape (47,)
```

**Concrete example (first 5 tokens from our sample file):**

| Index | Token ID | Decoded |
|-------|----------|---------|
| 0 | 40 | `"I"` |
| 1 | 1842 | `" love"` |
| 2 | 9552 | `" AI"` |
| 3 | 290 | `" and"` |
| 4 | 2769 | `" deep"` |

**Output shape:** `(N,)` where N = 47 for `sample.txt`.

---

### 5.3 `build_input_target_pairs(token_ids, block_size)`

**What it does:** Slide a window of size `T` over the long token list. Build two matrices `x` and `y`.

**Parameters:**

| Param | Example | Meaning |
|-------|---------|---------|
| `token_ids` | shape `(47,)` | entire corpus as one list |
| `block_size` | `T = 8` | tokens per training example |

**How many rows?**

```
num_blocks = N - T = 47 - 8 = 39 rows
```

**One row example (row 0, T=8):**

```
token_ids:  [40, 1842, 9552, 290, 2769, 4673, 13, 198, 41762, ...]
             └──────────── x[0] ────────────┘
                └──────────── y[0] ────────────┘

x[0] = token_ids[0:8]  = [40, 1842, 9552, 290, 2769, 4673, 13, 198]
y[0] = token_ids[1:9]  = [1842, 9552, 290, 2769, 4673, 13, 198, 41762]
```

**Decoded row 0:**

```
x[0]:  "I love AI and deep learning.\n"
y[0]:  " love AI and deep learning.\nTransform"
```

**Output:**

```
x.shape = (39, 8)
y.shape = (39, 8)
```

---

### 5.4 `verify_shift(x, y, token_ids, block_size)`

**What it does:** Debug check — confirms targets are truly "next token."

For row `b`, column `t`:

```
expected: token_ids[b + t + 1]
actual:   y[b, t]
```

If they don't match → `assert` crashes with an error message.

Only checks first 3 rows (fast sanity check, not full scan).

---

### 5.5 `create_dataloader(dataset, batch_size, shuffle)`

**What it does:** Wrap dataset so training can pull **batches** instead of single rows.

```python
dataset = TextTokenDataset(x, y)          # 39 examples, one row at a time
loader = create_dataloader(dataset, batch_size=4, shuffle=True)
```

With 39 examples and `batch_size=4`:

```
num_batches = ceil(39 / 4) = 10 batches
```

Last batch may have only 3 rows (because `drop_last=False`).

---

### 5.6 `train_val_split(dataset, val_fraction=0.1)`

**What it does:** Split 39 examples → ~35 train, ~4 validation.

```python
train_set, val_set = train_val_split(dataset, val_fraction=0.1)
len(train_set)  # 36
len(val_set)    # 3
```

Validation set is held out to measure how well the model generalizes (Phase 12).

---

## 6. The `TextTokenDataset` Class

### Full class (annotated)

```python
class TextTokenDataset(Dataset):
    # Inherit from Dataset → DataLoader knows how to use us

    def __init__(self, x, y):
        # Store the full (num_blocks, T) tables inside the object
        self.x = x
        self.y = y

    def __len__(self):
        # 39 rows → len(dataset) == 39
        return self.x.shape[0]

    def __getitem__(self, idx):
        # Return ONE row as (input_vector, target_vector)
        # Each vector shape (T,) — still 1D, NOT batched yet
        return self.x[idx], self.y[idx]
```

### Using the class manually (no DataLoader)

```python
dataset = TextTokenDataset(x, y)

print(len(dataset))        # 39

xi, yi = dataset[0]        # fetch row 0
print(xi.shape)            # (8,)
print(yi.shape)            # (8,)

xi, yi = dataset[7]        # fetch row 7 — a different sliding window
```

**At this point everything is still 1D per example.** No batch dimension yet.

---

## 7. Before Batching vs After Batching

This is the most important visual section.

### STAGE A — Raw text

```
sample.txt:

"I love AI and deep learning.
 Transformers learn language by predicting the next word.
 ..."
```

Just a string. No numbers. No tables.

---

### STAGE B — After `load_and_tokenize` → shape `(N,)`

One long **1D** vector (a single row, N columns):

```
token_ids  shape (47,):

index:  0    1     2     3    4     5        6   7    8      ...
      [40, 1842, 9552, 290, 2769, 4673,    13, 198, 41762, ...]
       I   love   AI   and  deep  learning  .   \n  Transform
```

Think: one very long sentence of numbers.

---

### STAGE C — After `build_input_target_pairs` → shape `(39, 8)`

Two **2D tables**. Each **row** is one training example. Still **no batching** — all 39 rows sit in memory at once.

```
x  shape (39, 8) — ALL rows stored:

row 0:  [40,  1842, 9552,  290, 2769, 4673,  13, 198]
row 1:  [1842, 9552,  290, 2769, 4673,   13, 198, 41762]
row 2:  [9552,  290, 2769, 4673,   13,  198, 41762, 364]
...
row 38: [... last window ...]

y  shape (39, 8) — same layout, shifted:

row 0:  [1842, 9552,  290, 2769, 4673,  13, 198, 41762]
row 1:  [9552,  290, 2769, 4673,   13,  198, 41762, 364]
...
```

---

### STAGE D — `dataset[3]` → one example, shape `(T,)`

DataLoader not involved. You pick **one row**:

```
dataset[3] returns:

xi = x[3]  shape (8,)   ← 1D vector, 8 numbers
yi = y[3]  shape (8,)   ← 1D vector, 8 numbers

xi: [one sliding window of 8 input tokens]
yi: [matching 8 target tokens]
```

**No batch dimension `B` yet.** Just one homework assignment from one student.

---

### STAGE E — After `DataLoader`, `batch_size=4` → shape `(B, T)`

DataLoader grabs **4 rows** and **stacks** them:

```
bx  shape (4, 8):

         col0  col1  col2  col3  col4  col5  col6  col7
row 0:  [  ..   ..    ..    ..    ..    ..    ..    ..  ]  ← was x[?] after shuffle
row 1:  [  ..   ..    ..    ..    ..    ..    ..    ..  ]
row 2:  [  ..   ..    ..    ..    ..    ..    ..    ..  ]
row 3:  [  ..   ..    ..    ..    ..    ..    ..    ..  ]

by  shape (4, 8):   same 4 rows from y, shifted targets
```

Now **`B = 4`** appears as the **first dimension**.

| Stage | Object | Shape | What dim 0 means | What dim 1 means |
|-------|--------|-------|------------------|------------------|
| B | `token_ids` | `(47,)` | — (1D only) | token index in corpus |
| C | `x`, `y` | `(39, 8)` | which example (row) | token position (col) |
| D | `dataset[i]` | `(8,)` | — (1D only) | token position |
| E | `bx`, `by` | `(4, 8)` | which example in batch | token position |

---

### Animation in text — one batch forming

```
Full x table (39 rows) — BEFORE batching, all rows visible:

    ROW 0
    ROW 1
    ROW 2
    ROW 3   ─┐
    ROW 4    │
    ...      │  DataLoader picks 4 rows (e.g. rows 3,7,1,20 if shuffled)
    ROW 38  ─┘

                ↓ stack into batch

bx shape (4, 8):

    [ ROW 3  ]
    [ ROW 7  ]
    [ ROW 1  ]
    [ ROW 20 ]
```

---

## 8. The `demo()` Function — Full Pipeline

`demo()` is the script you run. It calls everything in order:

| Step | Code | Output shape |
|------|------|--------------|
| 1 | `load_and_tokenize(data_path)` | `(47,)` |
| 2 | `build_input_target_pairs(..., T=8)` | `x:(39,8)`, `y:(39,8)` |
| 3 | print/decode row 0 | show "I love → love AI" pattern |
| 4 | `TextTokenDataset(x,y)` + `DataLoader` | batches `(4, 8)` |
| 5 | `train_val_split(...)` | 36 train, 3 val |

**Finding the data file:**

```python
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "sample.txt"
```

`__file__` = path to `phase01_dataset.py`
`.parents[2]` = go up 2 folders → project root (`SLM/`)

---

## 9. Reading the Terminal Output

When you run:

```bash
source .venv/bin/activate
python -m src.mini_gpt.phase01_dataset
```

You should see:

```
[load_and_tokenize] token_ids.shape = (47,)  (N,)
```
→ 47 tokens total in the file.

```
[build_input_target_pairs] x.shape = (39, 8)  (B, T)
[build_input_target_pairs] y.shape = (39, 8)  (B, T)
```
→ 39 training examples, each 8 tokens wide. (Here B means num_blocks, before DataLoader.)

```
bx.shape = (4, 8)  (B, T) — B=4 sequences, T=8 positions
```
→ Now B=4 because DataLoader grouped 4 rows.

```
bx[0] decoded: 'ers learn language by predicting the next word'
by[0] decoded: ' learn language by predicting the next word.'
```
→ Row 0 of the batch (random after shuffle) — still a coherent text chunk. Targets are shifted.

---

## 10. Quick Reference Cheat Sheet

| Code | Returns | Shape |
|------|---------|-------|
| `get_tokenizer()` | tokenizer object | — |
| `tokenizer.encode(text)` | list of ints | `(N,)` conceptually |
| `load_and_tokenize(path)` | LongTensor | `(N,)` |
| `build_input_target_pairs(ids, T)` | `x, y` | `(N-T, T)` each |
| `TextTokenDataset(x, y)` | dataset object | len = N-T |
| `dataset[i]` | `xi, yi` | `(T,)` each |
| `DataLoader(ds, batch_size=B)` | iterator | batches `(B, T)` |
| `train_val_split(ds)` | train, val | subsets |

---

## Common Code Questions

### Why `dtype=torch.long`?

Token IDs are integers (indices). `nn.Embedding` in Phase 2 requires integer indices, not floats.

### Why return `x, y` as a tuple?

So the caller can write `x, y = build_input_target_pairs(...)` — clean and readable.

### Why inherit `Dataset`?

`DataLoader` only works with objects that implement `__len__` and `__getitem__`. Inheriting `Dataset` documents that contract.

### Why `shuffle=True` only on rows?

Each row is one contiguous chunk of text. Shuffling **columns** would scramble word order inside a sentence and destroy meaning.

### What is `self`?

The specific object instance. `self.x` means "this dataset's x tensor."

---

## Next Step

Run the code yourself and match each print line to the stages in **Section 7**.

When ready, say **"go"** for Phase 2 (Token Embeddings) — where `(B, T)` becomes `(B, T, C)` with real number vectors.
