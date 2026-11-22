---
id: logistic-regression
module: classic
title: Logistic regression from scratch
minutes: 45
summary: A linear score, a sigmoid, a Bernoulli likelihood, and the gradient of that likelihood.
idea: Logistic regression is linear regression pointed at a probability.
---

## Intuition

A linear model produces a score $z = w^\top x + b$. Logistic regression squashes that score into a probability with the sigmoid, then treats the label as a Bernoulli draw with that probability. Training is maximum likelihood from the previous module, written as binary cross-entropy so that an optimizer can minimize it.

The gradient has a shape you should memorize. After the algebra, the error signal on each example is the predicted probability minus the label. That scalar multiplies the features to produce the gradient in $w$. There is no separate mystery derivative hiding in the sigmoid: it cancels with the derivative of the log loss, and $p - y$ is what remains. If your gradient check fails, compare against that expression before you touch the learning rate.

This lesson uses the mean loss, so the gradient is divided by the batch size. A sum loss is the same direction with a different scale, and people then hide the $1/n$ inside the learning rate. Pick one and stay with it. The tests use the mean.

The decision rule at 0.5 is a threshold on $z$ at 0, because the sigmoid of 0 is one half. The model is still linear in that sense: the boundary is a hyperplane. What is nonlinear is the probability, and that nonlinearity is exactly what makes the loss prefer confident correct answers over merely correct signs.

## Math

$$
\sigma(z) = \frac{1}{1+e^{-z}}, \qquad p_i = \sigma(w^\top x_i + b).
$$

For large positive $z$, compute $\sigma(z) = 1 / (1 + e^{-z})$. For large negative $z$, compute $\sigma(z) = e^{z} / (1 + e^{z})$ so the exponential never overflows.

Mean binary cross-entropy and its gradient:

$$
L = -\frac{1}{n}\sum_i \big( y_i \log p_i + (1-y_i)\log(1-p_i) \big),
$$

$$
\frac{\partial L}{\partial z_i} = \frac{p_i - y_i}{n}, \qquad
\nabla_w L = X^\top \frac{\partial L}{\partial z}, \qquad
\frac{\partial L}{\partial b} = \sum_i \frac{\partial L}{\partial z_i}.
$$

Clip probabilities only inside the loss, not inside the gradient. The gradient formula is already stable when you use a stable sigmoid.

## Resources

- [An Introduction to Statistical Learning](https://www.statlearning.com/) · book · James, Witten, Hastie, and Tibshirani · The free book's classification chapter is the statistical view of logistic regression.
- [Softmax regression](https://d2l.ai/chapter_linear-classification/softmax-regression.html) · reading · Dive into Deep Learning · The multiclass sibling, useful once the binary gradient feels routine.
- [CS229](https://cs229.stanford.edu/) · course · Stanford · The machine-learning course notes treat logistic regression as the first probabilistic classifier.
- [Deep Learning, chapter on machine learning basics](https://www.deeplearningbook.org/contents/ml.html) · book · Goodfellow, Bengio, and Courville · Likelihood, logistic regression, and the losses that follow from them.

## Exercise

Implement a stable sigmoid, probabilities, the mean negative log-likelihood, the mean gradients, and a batch training loop.

- `sigmoid(z)` handles large positive and negative inputs without overflow.
- `predict_proba(X, w, b)` returns a 1-D array of probabilities. `X` is `(n, d)`, `w` is `(d,)`.
- `logistic_nll(X, y, w, b)` returns the mean BCE as a Python float. Clip probabilities to `[1e-12, 1-1e-12]` inside this function only.
- `logistic_gradients(X, y, w, b)` returns `(grad_w, grad_b)` for the mean loss. `grad_b` is a Python float.
- `train_logistic(X, y, lr=0.5, steps=400)` starts at $w = 0$, $b = 0$. Return `(w, b, losses)` where `losses` has length `steps + 1`: the loss before any update, then the loss after each update.

```python starter
import numpy as np


def sigmoid(z):
    raise NotImplementedError


def predict_proba(X, w, b):
    raise NotImplementedError


def logistic_nll(X, y, w, b):
    raise NotImplementedError


def logistic_gradients(X, y, w, b):
    raise NotImplementedError


def train_logistic(X, y, lr=0.5, steps=400):
    raise NotImplementedError
```

```python solution
import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def predict_proba(X, w, b):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    return sigmoid(X @ w + b)


def logistic_nll(X, y, w, b):
    y = np.asarray(y, dtype=float)
    p = np.clip(predict_proba(X, w, b), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def logistic_gradients(X, y, w, b):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    residual = (predict_proba(X, w, b) - y) / X.shape[0]
    return X.T @ residual, float(np.sum(residual))


def train_logistic(X, y, lr=0.5, steps=400):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    losses = []
    for _ in range(int(steps)):
        losses.append(logistic_nll(X, y, w, b))
        grad_w, grad_b = logistic_gradients(X, y, w, b)
        w = w - lr * grad_w
        b = b - lr * grad_b
    losses.append(logistic_nll(X, y, w, b))
    return w, b, losses
```

```python tests
import numpy as np

z = np.array([-1000.0, 0.0, 1000.0])
s = sigmoid(z)
assert np.all(np.isfinite(s))
assert abs(s[1] - 0.5) < 1e-12
assert s[0] < 1e-6 and s[2] > 1 - 1e-6

rng = np.random.default_rng(1)
X = rng.normal(size=(80, 2))
labels = (1.5 * X[:, 0] - 1.2 * X[:, 1] + 0.3 > 0).astype(float)
w, b, losses = train_logistic(X, labels, lr=0.5, steps=400)
assert len(losses) == 401
assert losses[-1] < losses[0]
assert losses[-1] < 0.25
proba = predict_proba(X, w, b)
accuracy = np.mean((proba >= 0.5) == (labels == 1))
assert accuracy > 0.95

small_X = X[:8]
small_y = labels[:8]
w0 = np.array([0.2, -0.4])
b0 = 0.1
grad_w, grad_b = logistic_gradients(small_X, small_y, w0, b0)
eps = 1e-5
numeric_w = np.zeros_like(w0)
for i in range(w0.size):
    plus, minus = w0.copy(), w0.copy()
    plus[i] += eps
    minus[i] -= eps
    numeric_w[i] = (logistic_nll(small_X, small_y, plus, b0) - logistic_nll(small_X, small_y, minus, b0)) / (2 * eps)
numeric_b = (
    logistic_nll(small_X, small_y, w0, b0 + eps) - logistic_nll(small_X, small_y, w0, b0 - eps)
) / (2 * eps)
assert np.allclose(grad_w, numeric_w, atol=1e-5)
assert abs(grad_b - numeric_b) < 1e-5
```

## Quiz

### q1
After simplification, the derivative of binary cross-entropy with respect to the logit $z$ of one example is

- [ ] $y - \sigma(z)$
- [x] $\sigma(z) - y$
- [ ] $\sigma(z)\,(1-\sigma(z))$
- [ ] $z - y$

> The sigmoid derivative and the derivative of the log cancel. What remains is predicted probability minus label. The mean loss then divides by $n$.

### q2
A logistic model with $b = 0$ predicts probability 0.5 exactly when

- [ ] $\|x\| = 1$
- [ ] $y = 1$
- [x] $w^\top x = 0$
- [ ] the features have been z-scored

> $\sigma(0) = 1/2$. The set where the linear score is zero is the decision boundary.

### q3
Why clip probabilities inside the loss but keep the gradient as $(p-y)/n$?

- [ ] Clipping the gradient would make training faster on separable data.
- [x] A probability of 0 or 1 makes $\log p$ infinite, while $p-y$ is already finite if the sigmoid is stable.
- [ ] The two formulas are algebraically identical only after clipping.
- [ ] Gradient clipping is required for convex losses.

> The log is the fragile part. A stable sigmoid never returns a non-finite probability, and the simplified gradient does not take a log at all.
