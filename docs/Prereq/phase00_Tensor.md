## What does shape `(T,)` mean?

The notation **shape `(T,)`** means:

- It is a **1-dimensional array (vector)**.
- It contains **T elements**.
- There is **only one axis** (one dimension).

For example, if:

\[
T = 5
\]

and your tokens are:

```text
[10, 25, 7, 42, 19]
```

then:

```python
x.shape = (5,)
```

because there are 5 elements in a 1D vector.

---

## Compare with other shapes

### Shape `(T,)`

```text
[10, 25, 7, 42, 19]
```

One row of tokens.

Think:

```text
length = T
```

No separate row/column dimension.

---

### Shape `(1, T)`

```text
[[10, 25, 7, 42, 19]]
```

Now it is a **2D matrix**:

- 1 row
- T columns

Shape:

```python
(1, 5)
```

---

### Shape `(B, T)`

Suppose:

```text
B = 3
T = 5
```

```text
[
 [10, 25, 7, 42, 19],
 [ 3, 11, 9, 20, 15],
 [99, 12, 4,  8, 30]
]
```

Shape:

```python
(3, 5)
```

Meaning:

- 3 sequences (rows)
- 5 token positions per sequence (columns)

---

## In your GPT dataset

Suppose the token stream is:

```text
[5, 8, 2, 9, 1, 7, 4]
```

and:

```text
T = 4
```

Then one training example is:

```text
x = [5, 8, 2, 9]
y = [8, 2, 9, 1]
```

Both have:

```python
shape = (4,)
```

because each is just a single vector containing 4 token IDs.

---

## Why the comma?

In Python/NumPy/PyTorch:

```python
(4)
```

is just the number `4`.

But:

```python
(4,)
```

means a **tuple containing one element**.

Since shapes are stored as tuples:

```python
shape = (4,)
```

means:

> This tensor has one dimension whose size is 4.

Similarly:

```python
(3, 4)
```

means:

> This tensor has two dimensions: 3 rows and 4 columns.

---

## Memory Trick

| Shape | Meaning |
|---------|---------|
| `(T,)` | 1D vector of T tokens |
| `(1, T)` | 1 row, T columns |
| `(B, T)` | B sequences, each of length T |
| `(B, T, C)` | Batch × Positions × Features/Embeddings |

For GPT training:

- One sequence → `(T,)`
- Many sequences in a batch → `(B, T)`
- After embedding lookup → `(B, T, C)` where `C` is the embedding dimension.

---

## Visual Intuition

### Single sequence `(T,)`

```text
[ t0  t1  t2  t3 ]
```

Shape:

```python
(4,)
```

---

### Batch of sequences `(B, T)`

```text
[
 [t0 t1 t2 t3]
 [t4 t5 t6 t7]
 [t8 t9 t10 t11]
]
```

Shape:

```python
(3, 4)
```

- 3 sequences (`B = 3`)
- 4 tokens per sequence (`T = 4`)

---

### After Embedding `(B, T, C)`

Each token becomes a vector:

```text
[
 [
  [ ... C numbers ... ],
  [ ... C numbers ... ],
  [ ... C numbers ... ],
  [ ... C numbers ... ]
 ]
]
```

Shape:

```python
(B, T, C)
```

Example:

```python
(32, 128, 768)
```

means:

- Batch size = 32
- Sequence length = 128
- Embedding dimension = 768