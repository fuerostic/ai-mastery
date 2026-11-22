---
id: mlp-backprop
module: nets
title: Backprop through a small MLP
minutes: 40
summary: One hidden layer, ReLU, mean squared error, and every matrix the chain rule asks for.
idea: The scalar engine and the matrix engine are the same chain rule at two scales.
---

## Intuition

The previous lesson differentiated scalars. A layer differentiates arrays, and the arrays are outer structure around the same local rules. This network is one hidden layer:

$$
H = \mathrm{ReLU}(X W_1 + b_1), \qquad \hat Y = H W_2 + b_2.
$$

The loss is the mean of the squared entries of $\hat Y - Y$. Mean, not sum. That choice puts a factor of $2 / N$ on the output gradient, where $N$ is the number of entries in $\hat Y$. Drop the 2, or divide by the batch size instead of by $N$, and a finite-difference check will fail by a constant factor. Constant-factor bugs are the ones that still "train" and then confuse you.

Walk backward. The output gradient becomes a gradient for $W_2$ by the same outer product a linear regression would use: hidden activations transposed, times the output gradient. It becomes a gradient for the hidden layer by multiplying by $W_2^\top$. ReLU then masks that hidden gradient wherever the pre-activation was not strictly positive. The masked signal plays the role of the output gradient for the first linear layer.

Nothing here requires a framework. If these four matrices match a finite-difference probe, you understand the layer. The next lesson only wraps them in a loop.

## Math

$$
L = \frac{1}{N}\sum_{i,j} (\hat Y_{ij} - Y_{ij})^2, \qquad N = \hat Y.\mathrm{size}.
$$

$$
\frac{\partial L}{\partial \hat Y} = \frac{2}{N}(\hat Y - Y).
$$

$$
\begin{aligned}
\frac{\partial L}{\partial W_2} &= H^\top \frac{\partial L}{\partial \hat Y}, &
\frac{\partial L}{\partial b_2} &= \sum_{\text{rows}} \frac{\partial L}{\partial \hat Y}, \\
\frac{\partial L}{\partial H} &= \frac{\partial L}{\partial \hat Y} W_2^\top, &
\frac{\partial L}{\partial \mathrm{pre}} &= \frac{\partial L}{\partial H} \odot [\mathrm{pre} > 0].
\end{aligned}
$$

Then $W_1$ and $b_1$ are the linear-layer gradients of that masked signal against $X$. At a pre-activation of exactly zero the ReLU gate is 0.

Shapes, with batch $n$, input $d$, hidden $h$, output $c$: $X$ is $(n,d)$, $W_1$ is $(d,h)$, $b_1$ is $(h,)$, $W_2$ is $(h,c)$, $b_2$ is $(c,)$, $Y$ is $(n,c)$.

## Resources

- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) · video · Andrej Karpathy · The backpropagation lecture builds this layer after the scalar engine.
- [Neural Networks and Deep Learning, chapter 2](http://neuralnetworksanddeeplearning.com/chap2.html) · book · Michael Nielsen · The index form of the same derivatives.
- [Multilayer perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) · reading · Dive into Deep Learning · The architecture, before and after the derivatives.
- [CS231n: neural networks, part 1](https://cs231n.github.io/neural-networks-1/) · reading · Stanford CS231n · Layers, activations, and losses in one set of notes.
- [But what is a neural network?](https://www.3blue1brown.com/topics/neural-networks) · video · 3Blue1Brown · The geometric picture of layers as maps, useful before the index chase.

## Exercise

Implement the forward pass, the mean squared loss, and the backward pass.

- `mlp_forward(X, W1, b1, W2, b2)` returns `(yhat, cache)` with `cache = (X, pre, H, yhat)`. `pre` is the hidden pre-activation. `H` is ReLU of `pre`.
- `mlp_loss(yhat, y)` returns the mean squared error as a Python float.
- `mlp_gradients(cache, y, W2)` returns `(dW1, db1, dW2, db2)`. Use the factor $2/N$ and a strict `pre > 0` gate.

A hand example worth trusting before the finite-difference test: $X = [[1]]$, $W_1 = [[1, -1]]$, $b_1 = [0, 1]$, $W_2 = [[1], [2]]$, $b_2 = [0.5]$, $y = [[2]]$. Then `yhat` is $[[1.5]]$, the loss is $0.25$, `dW2` is $[[-1], [0]]$, `db2` is $[-1]$, `dW1` is $[[-1, 0]]$, and `db1` is $[-1, 0]$.

```python starter
import numpy as np


def mlp_forward(X, W1, b1, W2, b2):
    raise NotImplementedError


def mlp_loss(yhat, y):
    raise NotImplementedError


def mlp_gradients(cache, y, W2):
    raise NotImplementedError
```

```python solution
import numpy as np


def mlp_forward(X, W1, b1, W2, b2):
    X = np.asarray(X, dtype=float)
    pre = X @ np.asarray(W1, dtype=float) + np.asarray(b1, dtype=float)
    hidden = np.maximum(pre, 0.0)
    yhat = hidden @ np.asarray(W2, dtype=float) + np.asarray(b2, dtype=float)
    return yhat, (X, pre, hidden, yhat)


def mlp_loss(yhat, y):
    yhat = np.asarray(yhat, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((yhat - y) ** 2))


def mlp_gradients(cache, y, W2):
    X, pre, hidden, yhat = cache
    y = np.asarray(y, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    d_yhat = 2.0 * (yhat - y) / yhat.size
    dW2 = hidden.T @ d_yhat
    db2 = d_yhat.sum(axis=0)
    d_hidden = d_yhat @ W2.T
    d_pre = d_hidden * (pre > 0)
    dW1 = X.T @ d_pre
    db1 = d_pre.sum(axis=0)
    return dW1, db1, dW2, db2
```

```python tests
import numpy as np

X = np.array([[1.0]])
W1 = np.array([[1.0, -1.0]])
b1 = np.array([0.0, 1.0])
W2 = np.array([[1.0], [2.0]])
b2 = np.array([0.5])
y = np.array([[2.0]])
yhat, cache = mlp_forward(X, W1, b1, W2, b2)
assert np.allclose(yhat, np.array([[1.5]]))
assert abs(mlp_loss(yhat, y) - 0.25) < 1e-12
dW1, db1, dW2, db2 = mlp_gradients(cache, y, W2)
assert np.allclose(dW2, np.array([[-1.0], [0.0]]))
assert np.allclose(db2, np.array([-1.0]))
assert np.allclose(dW1, np.array([[-1.0, 0.0]]))
assert np.allclose(db1, np.array([-1.0, 0.0]))

rng = np.random.default_rng(2)
X = rng.normal(size=(6, 3))
y = rng.normal(size=(6, 2))
W1 = rng.normal(size=(3, 4))
b1 = rng.normal(size=4)
W2 = rng.normal(size=(4, 2))
b2 = rng.normal(size=2)
yhat, cache = mlp_forward(X, W1, b1, W2, b2)
dW1, db1, dW2, db2 = mlp_gradients(cache, y, W2)
eps = 1e-5

def loss_of(a, c, e, g):
    pred, _ = mlp_forward(X, a, c, e, g)
    return mlp_loss(pred, y)

numeric = np.zeros_like(W1)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        plus, minus = W1.copy(), W1.copy()
        plus[i, j] += eps
        minus[i, j] -= eps
        numeric[i, j] = (loss_of(plus, b1, W2, b2) - loss_of(minus, b1, W2, b2)) / (2 * eps)
assert np.allclose(dW1, numeric, atol=1e-5)
numeric = np.zeros_like(W2)
for i in range(W2.shape[0]):
    for j in range(W2.shape[1]):
        plus, minus = W2.copy(), W2.copy()
        plus[i, j] += eps
        minus[i, j] -= eps
        numeric[i, j] = (loss_of(W1, b1, plus, b2) - loss_of(W1, b1, minus, b2)) / (2 * eps)
assert np.allclose(dW2, numeric, atol=1e-5)
```

## Quiz

### q1
The loss is the mean of squared residuals over every entry of $\hat Y$. The output gradient therefore starts with

- [ ] $\hat Y - Y$
- [x] $2(\hat Y - Y) / N$, where $N$ is the number of entries
- [ ] $2(\hat Y - Y) / n$, where $n$ is the batch size, even if there are many outputs
- [ ] $\mathrm{softmax}(\hat Y) - Y$

> Differentiating $u^2$ produces $2u$, and the mean divides by the number of terms in the mean. If $\hat Y$ has several columns, $N$ is larger than the batch size.

### q2
ReLU's backward mask in this lesson is `pre > 0`, not `hidden > 0`. Those differ when

- [ ] the batch size is 1.
- [x] a pre-activation is exactly zero. Both the value and the gate are then zero, but the comparison should be made on the pre-activation.
- [ ] $W_2$ has a negative entry.
- [ ] they never differ. ReLU is idempotent.

> For a strictly positive or strictly negative pre-activation the two comparisons agree. The tie at zero is defined on the pre-activation, which is the input of ReLU.

### q3
`dW2 = H.T @ dY` is the gradient of a linear layer because

- [ ] the transpose converts a row vector into a probability.
- [x] each weight multiplies one hidden unit and one output, so its derivative is that hidden activation times the output gradient.
- [ ] mean squared error requires an extra Jacobian the first layer does not.
- [ ] $H$ is orthogonal.

> That is the outer product of the layer's input with the gradient of its output, summed over the batch by the matrix product.
