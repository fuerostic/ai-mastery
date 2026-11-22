---
id: numpy-thinking
module: classic
title: Thinking in array shapes
minutes: 30
summary: Say the shape of every intermediate before you write the line.
idea: Broadcasting is a disciplined way to avoid a Python loop, not a license to guess.
---

## Intuition

A Python loop hides the math under indexes. NumPy lets the formula look like the formula, but only if you can say the shape of each piece. "Subtract the column mean" is an `(n, d)` matrix minus a `(d,)` vector. Broadcasting stretches the vector down the rows. It does not invent a new axis unless you ask, and it refuses to guess when two dimensions disagree and neither is 1.

Three operations cover most of classical learning code. Centering and scaling a matrix by columns. A row-wise softmax, which has to subtract the row max before the exponential or a logit of 1000 becomes `inf`. A one-hot encoding, which turns integer labels into the vectors cross-entropy expects.

The failure mode is a silent shape. `(n, d)` times `(n, d)` elementwise is legal and usually wrong if you meant a matrix product. Read the line as shapes first and numbers second. If you cannot, you are guessing, and the test will eventually notice.

## Math

Column z-scoring uses the population standard deviation, the one that divides by $n$:

$$
Z_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j}, \qquad
\sigma_j^2 = \frac{1}{n}\sum_i (X_{ij}-\mu_j)^2.
$$

If a column has $\sigma_j = 0$, this lesson sets that column of $Z$ to 0 after centering. Dividing by zero is not a result you want to train on.

A stable row softmax subtracts the max inside the row:

$$
\mathrm{softmax}(z)_j = \frac{e^{z_j - \max_k z_k}}{\sum_i e^{z_i - \max_k z_k}}.
$$

Rows sum to 1. One-hot encoding of an index $y$ in $\{0,\ldots,C-1\}$ is the vector $e_y$ with a 1 in position $y$.

## Resources

- [NumPy: the absolute basics](https://numpy.org/doc/stable/user/absolute_beginners.html) · reading · NumPy · Arrays, shapes, and the operations this lesson assumes you can write.
- [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) · reading · NumPy · The actual rule: dimensions are compatible when they are equal or one of them is 1.
- [Python NumPy tutorial](https://cs231n.github.io/python-numpy-tutorial/) · reading · Stanford CS231n · The short tutorial written for people who are about to implement learning algorithms.
- [Linear algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html) · reading · Dive into Deep Learning · The same shapes, shown next to the deep-learning code that uses them.

## Exercise

Implement three functions. Prefer vectorized NumPy. A correct loop still passes, but say the shapes anyway.

- `zscore_columns(X)` takes `(n, d)` and returns `(n, d)`. Use the population standard deviation (`ddof=0`). A constant column becomes all zeros.
- `row_softmax(logits)` takes `(n, c)` and returns `(n, c)`. Subtract the per-row maximum before `exp`. Each row sums to 1.
- `one_hot(indices, n_classes)` takes a 1-D integer array and returns a float array of shape `(len(indices), n_classes)`.

```python starter
import numpy as np


def zscore_columns(X):
    raise NotImplementedError


def row_softmax(logits):
    raise NotImplementedError


def one_hot(indices, n_classes):
    raise NotImplementedError
```

```python solution
import numpy as np


def zscore_columns(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    constant = std == 0.0
    safe = np.where(constant, 1.0, std)
    scaled = (X - mean) / safe
    scaled[:, constant] = 0.0
    return scaled


def row_softmax(logits):
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(indices, n_classes):
    indices = np.asarray(indices, dtype=int)
    out = np.zeros((indices.shape[0], int(n_classes)), dtype=float)
    out[np.arange(indices.shape[0]), indices] = 1.0
    return out
```

```python tests
import numpy as np

X = np.array([[1.0, 5.0], [3.0, 5.0]])
Z = zscore_columns(X)
assert np.allclose(Z, np.array([[-1.0, 0.0], [1.0, 0.0]]))
constant = np.ones((4, 1))
assert np.allclose(zscore_columns(constant), np.zeros((4, 1)))

logits = np.array([[1000.0, 1001.0], [0.0, 0.0]])
probs = row_softmax(logits)
assert np.all(np.isfinite(probs))
assert np.allclose(probs.sum(axis=1), np.ones(2))
assert probs[0, 1] > 0.7
assert np.allclose(probs[1], np.array([0.5, 0.5]))

encoded = one_hot(np.array([0, 2, 1]), 3)
assert np.allclose(encoded, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float))
```

## Quiz

### q1
`X` has shape `(32, 10)` and `mu = X.mean(axis=0)`. What is the shape of `X - mu`?

- [ ] `(32,)`
- [ ] `(10, 32)`
- [x] `(32, 10)`
- [ ] The subtraction is illegal without `reshape`.

> A length-10 vector broadcasts down the rows of a `(32, 10)` matrix. Averaging over axis 0 is what made the vector length 10.

### q2
Why subtract the row max before exponentiating a softmax?

- [ ] It makes the largest probability exactly 1.
- [x] It keeps the exponentials finite. Subtracting a constant does not change the softmax.
- [ ] It converts the softmax into a sigmoid.
- [ ] It is only required when the number of classes is two.

> $e^{z_j - c} / \sum_i e^{z_i - c} = e^{z_j} / \sum_i e^{z_i}$. Choosing $c = \max z$ makes the largest exponential equal to 1.

### q3
A one-hot vector for class 2, with 4 classes, is

- [ ] `(0.25, 0.25, 0.25, 0.25)`
- [ ] `(2, 0, 0, 0)`
- [x] `(0, 0, 1, 0)`
- [ ] `(0, 1, 0, 0)`

> Class indexes start at 0, so class 2 is the third coordinate. The vector is an indicator, not a probability and not the integer label.
