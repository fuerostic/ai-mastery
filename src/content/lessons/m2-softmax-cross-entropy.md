---
id: softmax-cross-entropy
module: classic
title: Softmax and multiclass cross-entropy
minutes: 40
summary: Many classes, one normalized distribution, and a gradient that is the prediction minus the target.
idea: Softmax is the multiclass sigmoid, and its loss gradient is almost embarrassingly simple.
---

## Intuition

Logistic regression had two outcomes and one probability. Multiclass classification has $C$ scores and needs a distribution over them. Softmax exponentiates the scores and divides by their sum. The largest score wins the most probability, and the gaps between scores, not their absolute level, decide the distribution. Adding 10 to every logit changes nothing. That invariance is why a stable implementation subtracts the max before exponentiating.

Cross-entropy against a one-hot label is the negative log of the probability on the correct class. It is again a negative log-likelihood, now of a categorical distribution. The gradient with respect to the logits is the softmax probability vector with 1 subtracted on the correct class, divided by the batch size if the loss is a mean. Rows of that gradient sum to zero: softmax outputs live on a simplex, and the loss cannot push every class up at once.

You will see this pair again as the pretraining loss of a language model. The "classes" there are the tokens in the vocabulary, and the correct class is the next token. Nothing about the gradient changes. Only $C$ gets large.

## Math

For a row of logits $z \in \mathbb{R}^C$,

$$
\mathrm{softmax}(z)_j = \frac{e^{z_j - m}}{\sum_k e^{z_k - m}}, \qquad m = \max_k z_k.
$$

Mean categorical cross-entropy for a batch of logits $Z \in \mathbb{R}^{n \times C}$ and integer targets $y_i$ is

$$
L = -\frac{1}{n}\sum_{i=1}^n \log \mathrm{softmax}(Z_i)_{y_i}.
$$

The gradient in the logits is

$$
\frac{\partial L}{\partial Z_i} = \frac{1}{n}\big(\mathrm{softmax}(Z_i) - e_{y_i}\big).
$$

Compute the log-softmax as $z_j - m - \log \sum_k e^{z_k - m}$ when you need the loss itself. Materializing the probabilities and then taking a log is a good way to underflow.

## Resources

- [Softmax regression](https://d2l.ai/chapter_linear-classification/softmax-regression.html) · reading · Dive into Deep Learning · The loss, the gradient, and a training loop in the same notation.
- [Information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html) · reading · Dive into Deep Learning · Why cross-entropy is the right name for $-\sum p \log q$.
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) · video · Andrej Karpathy · The makemore lectures derive this loss for a character-level model. Watch them next to the network module.
- [CS231n: neural networks, part 1](https://cs231n.github.io/neural-networks-1/) · reading · Stanford CS231n · Softmax and SVM losses compared on the same linear scores.

## Exercise

Implement a stable softmax, the mean cross-entropy, and the logit gradient.

- `stable_softmax(logits)` accepts a 1-D or 2-D array and normalizes over the last axis. Outputs are finite and sum to 1 along that axis.
- `categorical_cross_entropy(logits, targets)` takes logits shaped `(n, C)` and integer targets shaped `(n,)`. Return the mean negative log-probability as a Python float. Use a log-sum-exp, not `log(softmax(...))`.
- `softmax_grad(logits, targets)` returns $\partial L / \partial \text{logits}$ for that mean loss, with the same shape as `logits`.

```python starter
import numpy as np


def stable_softmax(logits):
    raise NotImplementedError


def categorical_cross_entropy(logits, targets):
    raise NotImplementedError


def softmax_grad(logits, targets):
    raise NotImplementedError
```

```python solution
import numpy as np


def stable_softmax(logits):
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def categorical_cross_entropy(logits, targets):
    z = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)
    z = z - np.max(z, axis=-1, keepdims=True)
    log_z = z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))
    n = z.shape[0]
    return float(-np.mean(log_z[np.arange(n), targets]))


def softmax_grad(logits, targets):
    probs = stable_softmax(logits)
    targets = np.asarray(targets, dtype=int)
    n = probs.shape[0]
    probs = probs.copy()
    probs[np.arange(n), targets] -= 1.0
    return probs / n
```

```python tests
import numpy as np

pair = stable_softmax(np.array([0.0, 0.0]))
assert np.allclose(pair, np.array([0.5, 0.5]))
huge = stable_softmax(np.array([[1000.0, 1001.0], [-5.0, -5.0]]))
assert np.all(np.isfinite(huge))
assert np.allclose(huge.sum(axis=-1), np.ones(2))
assert huge[0, 1] > 0.7

assert abs(categorical_cross_entropy(np.array([[0.0, 0.0]]), np.array([0])) - np.log(2)) < 1e-9
assert abs(categorical_cross_entropy(np.array([[0.0, 0.0]]), np.array([1])) - np.log(2)) < 1e-9

batch = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
targets = np.array([2, 0])
grad = softmax_grad(batch, targets)
assert grad.shape == batch.shape
assert np.allclose(grad.sum(axis=1), np.zeros(2), atol=1e-8)

eps = 1e-5
numeric = np.zeros_like(batch)
for i in range(batch.shape[0]):
    for j in range(batch.shape[1]):
        plus, minus = batch.copy(), batch.copy()
        plus[i, j] += eps
        minus[i, j] -= eps
        numeric[i, j] = (
            categorical_cross_entropy(plus, targets) - categorical_cross_entropy(minus, targets)
        ) / (2 * eps)
assert np.allclose(grad, numeric, atol=1e-5)
```

## Quiz

### q1
Adding the same constant to every logit of one example changes the softmax by

- [ ] multiplying every probability by that constant.
- [ ] shifting the mode to a different class whenever the constant is larger than 1.
- [x] nothing. The probabilities are unchanged.
- [ ] making the distribution uniform.

> The constant factors out of every exponential and cancels in the fraction. Only differences between logits matter.

### q2
For a mean cross-entropy loss, the gradient with respect to the logits of one example is

- [ ] the one-hot target, divided by $n$.
- [x] $(\mathrm{softmax}(z) - e_y) / n$
- [ ] $\mathrm{softmax}(z) \odot (1 - \mathrm{softmax}(z))$
- [ ] the logits themselves.

> Same cancellation as in logistic regression. The predicted distribution moves toward the one-hot target, and the $1/n$ belongs to the mean.

### q3
Why do the logit-gradients of one example sum to zero?

- [ ] Because the loss is convex.
- [ ] Because every target is class 0 in the tests.
- [x] Because softmax probabilities sum to 1, and subtracting the one-hot target removes exactly 1.
- [ ] They do not. That would mean the loss has no minimum.

> $\sum_j p_j - 1 = 0$. The simplex constraint is visible in the gradient: you cannot increase every class's logit-gradient at once.
