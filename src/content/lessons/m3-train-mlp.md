---
id: train-mlp
module: nets
title: Training an MLP on a problem a line cannot fit
minutes: 40
summary: XOR is the smallest dataset that forces the hidden layer to earn its keep.
idea: A hidden layer is only useful if you can point at a dataset a linear model misses.
---

## Intuition

Logistic regression draws one hyperplane. The four XOR points do not have one. Points $(0,0)$ and $(1,1)$ are the negative class, and the other two corners are positive. Any straight cut leaves a corner on the wrong side. A hidden layer of ReLUs folds the space first. After that fold, a second linear map can separate the classes.

Training is the previous lesson inside a loop. Initialize the weights at a sensible scale, compute the loss, compute the four gradients, step, repeat. He initialization sets the standard deviation of a ReLU layer to $\sqrt{2 / \mathrm{fan\_in}}$ so that the forward activations do not systematically shrink or explode. Biases start at zero. The learning rate is a constant in this exercise. XOR is small enough that a constant rate, a wide enough hidden layer, and a few hundred steps are plenty.

Watch the loss, not a story about the loss. It should fall by a lot, and the four predictions should land on the correct side of 0.5. If the loss stalls near the mean of a constant prediction, the hidden layer is not moving, which usually means a zero gradient from a dead ReLU or a step size that left the useful region on the first update.

This is still not a framework. There is no batching strategy, no regularization, and no validation set. The point is the closed loop: forward, loss, backward, step. Every later training lesson is that loop with a different loss.

## Math

He normal initialization for a weight matrix with fan-in $d$:

$$
W_{ij} \sim \mathcal{N}\!\left(0, \sqrt{\frac{2}{d}}\right).
$$

The update, with learning rate $\eta$, is the gradient step from the optimization lesson:

$$
W \leftarrow W - \eta \frac{\partial L}{\partial W}, \qquad b \leftarrow b - \eta \frac{\partial L}{\partial b}.
$$

XOR as a regression target, not a classification loss, is enough here. Targets are 0 or 1 and the output is linear, so the network can predict numbers outside $[0,1]$. The decision rule used in the test is still a threshold at one half.

## Resources

- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) · video · Andrej Karpathy · The makemore series trains tiny networks on purpose, so you can see the loss move.
- [makemore](https://github.com/karpathy/makemore) · code · Andrej Karpathy · Character-level models that start from the same MLP you are training here.
- [Multilayer perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) · reading · Dive into Deep Learning · Capacity, hidden layers, and the XOR-shaped motivation.
- [Practical Deep Learning for Coders](https://course.fast.ai/) · course · fast.ai · A full course on training loops once the scratch version feels small.
- [A recipe for training neural networks](http://karpathy.github.io/2019/04/25/recipe/) · reading · Andrej Karpathy · How to tell a dead network from an unlucky one.

## Exercise

Using `mlp_forward`, `mlp_loss`, and `mlp_gradients` from the previous lesson's contracts, implement a trainer and a predictor. You may redefine those three functions in this file. The tests call `train_mlp` and `mlp_predict` only.

- `train_mlp(X, y, hidden=8, lr=0.1, steps=200, seed=0)` initializes `W1` and `W2` with `np.random.default_rng(seed)` using He normal scales, and biases at zero. `y` has shape `(n, c)`. Return `(W1, b1, W2, b2, losses)` with one loss per step, recorded before the update. `losses` has length `steps`.
- `mlp_predict(X, W1, b1, W2, b2)` returns $\hat Y$.

The tests fit XOR with `hidden=16`, `lr=0.4`, `steps=500`, `seed=0`. Your loop has to honor the arguments rather than hard-code those values.

```python starter
import numpy as np


def mlp_forward(X, W1, b1, W2, b2):
    raise NotImplementedError


def mlp_loss(yhat, y):
    raise NotImplementedError


def mlp_gradients(cache, y, W2):
    raise NotImplementedError


def train_mlp(X, y, hidden=8, lr=0.1, steps=200, seed=0):
    raise NotImplementedError


def mlp_predict(X, W1, b1, W2, b2):
    raise NotImplementedError
```

```python solution
import numpy as np


def mlp_forward(X, W1, b1, W2, b2):
    pre = np.asarray(X, dtype=float) @ W1 + b1
    hidden = np.maximum(pre, 0.0)
    yhat = hidden @ W2 + b2
    return yhat, (np.asarray(X, dtype=float), pre, hidden, yhat)


def mlp_loss(yhat, y):
    return float(np.mean((yhat - y) ** 2))


def mlp_gradients(cache, y, W2):
    X, pre, hidden, yhat = cache
    d_yhat = 2.0 * (yhat - y) / yhat.size
    dW2 = hidden.T @ d_yhat
    db2 = d_yhat.sum(axis=0)
    d_pre = (d_yhat @ W2.T) * (pre > 0)
    dW1 = X.T @ d_pre
    db1 = d_pre.sum(axis=0)
    return dW1, db1, dW2, db2


def train_mlp(X, y, hidden=8, lr=0.1, steps=200, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    n, d = X.shape
    out = y.shape[1]
    W1 = rng.normal(scale=np.sqrt(2.0 / d), size=(d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(scale=np.sqrt(2.0 / hidden), size=(hidden, out))
    b2 = np.zeros(out)
    losses = []
    for _ in range(int(steps)):
        yhat, cache = mlp_forward(X, W1, b1, W2, b2)
        losses.append(mlp_loss(yhat, y))
        dW1, db1, dW2, db2 = mlp_gradients(cache, y, W2)
        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2
    return W1, b1, W2, b2, losses


def mlp_predict(X, W1, b1, W2, b2):
    yhat, _cache = mlp_forward(X, W1, b1, W2, b2)
    return yhat
```

```python tests
import numpy as np

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])
W1, b1, W2, b2, losses = train_mlp(X, y, hidden=16, lr=0.4, steps=500, seed=0)
assert len(losses) == 500
assert losses[-1] < losses[0]
assert losses[-1] < 0.02
pred = mlp_predict(X, W1, b1, W2, b2)
assert np.array_equal((pred >= 0.5).astype(float), y)

again = train_mlp(X, y, hidden=16, lr=0.4, steps=5, seed=0)
assert np.allclose(again[0], train_mlp(X, y, hidden=16, lr=0.4, steps=5, seed=0)[0])
```

## Quiz

### q1
Why can a one-hidden-layer ReLU network fit XOR when logistic regression cannot?

- [ ] ReLU is a probability, so the loss becomes convex.
- [ ] XOR is linearly separable once you add a bias.
- [x] The hidden layer warps the four points so that a later linear boundary can separate them.
- [ ] Mean squared error ignores the negative class.

> The positive XOR points are not a half-space. A nonlinear feature map, even a shallow one, can make them one.

### q2
He initialization uses scale $\sqrt{2/\mathrm{fan\_in}}$ because

- [ ] that is the learning rate Adam will assume.
- [x] a ReLU zeros about half of a symmetric pre-activation, and the factor 2 puts the surviving variance back near 1.
- [ ] biases must start at that scale too.
- [ ] XOR requires integer weights.

> Without the 2, each ReLU layer would tend to shrink activation variance. The next lesson measures a related failure for sigmoids.

### q3
The returned `losses` list has one entry per step, recorded

- [x] before the parameter update, so `losses[0]` is the loss at initialization.
- [ ] after the update, so it always has length `steps + 1`.
- [ ] only when the loss decreased.
- [ ] on the training set and the test set interleaved.

> Recording first makes the curve start at the untrained model. The length equals `steps`, not `steps + 1`.
