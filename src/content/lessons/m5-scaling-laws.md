---
id: scaling-laws
module: training
title: Scaling laws
minutes: 35
summary: Loss falls as a power of scale, until you spend the compute on the wrong axis.
idea: A straight line on a log-log plot is the whole empirical claim.
---

## Intuition

People noticed that, over a wide range, the cross-entropy of a language model falls smoothly as you add parameters, data, and compute. Not in mysterious jumps. As a power law. Plot $\log \mathrm{loss}$ against $\log N$ and you see a line. That is a scaling law. It is an empirical fit, not a theorem, and it has been revised. The useful part is the discipline: before you invent a new architecture, ask where a plain transformer of this size ought to land.

Kaplan and colleagues fit loss against parameters, data, and compute, and the parameter axis got most of the attention. Hoffmann and colleagues, in the Chinchilla paper, argued that many large models were undertrained. For a fixed compute budget there is a best split between model size and number of tokens, and that split gave smaller models more data than the earlier fashion did. Both papers are linked below. Read the plots before the appendices.

This lesson fits the simplest version, a parameter-only power law $L(N) \approx a N^{-b}$, with no irreducible entropy term. Real fits add a floor, because even a perfect model cannot drive natural-language cross-entropy to zero. Leaving the floor out will bias $a$ and $b$ if your losses are close to that floor. On a clean synthetic curve the two-parameter fit recovers the constants you planted. That is the exercise. The judgment about whether a published curve used the right x-axis is the reading.

## Math

Take logarithms:

$$
\log L = \log a - b \log N.
$$

That is a straight line in $(\log N, \log L)$ with slope $-b$ and intercept $\log a$. Ordinary least squares recovers both. Prediction is $a N^{-b}$.

$N$ and $L$ must be positive. Two points determine a line. More points, and the fit is the least-squares compromise. This is not a claim that the world is a power law. It is the regression the papers start from.

## Resources

- [Scaling laws for neural language models](https://arxiv.org/abs/2001.08361) · paper · Kaplan et al. · The power-law fits against parameters, data, and compute.
- [Training compute-optimal large language models](https://arxiv.org/abs/2203.15556) · paper · Hoffmann et al. · The Chinchilla result: many models were too big for the data they saw.
- [Language models are few-shot learners](https://arxiv.org/abs/2005.14165) · paper · Brown et al. · GPT-3, and the scaling curves that made the Kaplan-style plots famous.
- [Stanford CS336](https://stanford-cs336.github.io/) · course · Stanford · Language modeling from scratch, including how these empirical laws are used when you actually train.
- [LLMs from scratch, chapter 5](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05) · code · Sebastian Raschka · The pretraining loop whose loss a scaling law is trying to predict.

## Exercise

Fit and evaluate $L \approx a N^{-b}$.

- `fit_scaling_law(n_params, losses)` takes two 1-D sequences of positive numbers, with at least two points. Fit $\log L = \log a - b \log N$ by least squares. Return `(a, b)` as Python floats. Raise `ValueError` if the inputs are not the same length, have fewer than two points, or contain a non-positive number.
- `predict_scaling_loss(n_params, a, b)` returns $a \cdot n^{-b}$ as a Python float.

`np.polyfit` is a fair way to fit the line. The slope it returns is $-b$, not $b$.

```python starter
import numpy as np


def fit_scaling_law(n_params, losses):
    raise NotImplementedError


def predict_scaling_loss(n_params, a, b):
    raise NotImplementedError
```

```python solution
import numpy as np


def fit_scaling_law(n_params, losses):
    n_params = np.asarray(n_params, dtype=float)
    losses = np.asarray(losses, dtype=float)
    if n_params.ndim != 1 or losses.shape != n_params.shape or n_params.size < 2:
        raise ValueError("n_params and losses must be 1-D and the same length, with at least two points")
    if np.any(n_params <= 0) or np.any(losses <= 0):
        raise ValueError("n_params and losses must be positive")
    slope, intercept = np.polyfit(np.log(n_params), np.log(losses), 1)
    return float(np.exp(intercept)), float(-slope)


def predict_scaling_loss(n_params, a, b):
    return float(a * np.asarray(n_params, dtype=float) ** (-b))
```

```python tests
import numpy as np

sizes = np.array([1e2, 1e3, 1e4, 1e5])
losses = 2.0 * sizes ** (-0.5)
a, b = fit_scaling_law(sizes, losses)
assert abs(a - 2.0) < 1e-6
assert abs(b - 0.5) < 1e-6
assert abs(predict_scaling_loss(1e6, a, b) - 2.0 * 1e6 ** (-0.5)) < 1e-6
try:
    fit_scaling_law([1.0, 2.0], [0.3])
    raise AssertionError("mismatched lengths should be rejected")
except ValueError:
    pass
try:
    fit_scaling_law([1.0, -2.0], [0.3, 0.2])
    raise AssertionError("non-positive inputs should be rejected")
except ValueError:
    pass
```

## Quiz

### q1
On a log-log plot, $L = a N^{-b}$ is

- [ ] a parabola.
- [x] a straight line with slope $-b$.
- [ ] a straight line with slope $a$.
- [ ] undefined, because a power law is not differentiable.

> $\log L = \log a - b \log N$. The slope is the exponent, with a minus sign because loss falls as $N$ grows.

### q2
The Chinchilla result changed the practical reading of scaling laws by saying that

- [ ] parameters do not matter once the model is deeper than 12 layers.
- [x] for a fixed compute budget, earlier large models had too few training tokens relative to their size.
- [ ] cross-entropy is the wrong pretraining loss.
- [ ] power laws only hold for vision models.

> Compute can buy parameters or tokens. The compute-optimal frontier spends more on tokens than the models that were chasing parameter count alone.

### q3
This lesson's fit has no irreducible term $L_\infty$. That omission matters when

- [ ] $N$ is a power of ten.
- [x] the measured losses are close to the entropy floor of the data, so a pure power law bends.
- [ ] you use natural logs instead of log base 10.
- [ ] there are more than two points.

> A pure power law heads toward zero. Language does not. Near the floor, the two-parameter curve is the wrong function, which is why the papers include a constant.
