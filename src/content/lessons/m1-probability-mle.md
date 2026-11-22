---
id: probability-mle
module: math
title: Likelihood, cross-entropy, and a bootstrap interval
minutes: 45
summary: Fit a probability model by making the data unsurprising, then ask how stable the average is.
idea: Cross-entropy is negative log-likelihood wearing a machine-learning name.
---

## Intuition

A probabilistic model is a story that assigns a probability to the data. The likelihood is that same number read as a function of the parameters, with the data held fixed. Maximum likelihood picks the story under which the observations look least surprising. For a coin, the story is the obvious one: the heads probability equals the fraction of heads you saw. For a Gaussian with unknown mean and variance, the mean is the sample average and the MLE variance divides by $n$, not by $n-1$. The $n-1$ version is a different estimator, with a different purpose.

Products of probabilities become sums when you take a log, and a minus sign turns maximization into the minimization that optimizers expect. Negative log-likelihood is the training loss of a probabilistic model. Binary cross-entropy is exactly the average negative log-likelihood of a Bernoulli whose success probability is your prediction $p$. Multiclass cross-entropy, in the next module, is the same idea for a softmax.

A single average is a random variable. Draw the test set again and the number moves. The bootstrap is a way to see that movement without a new experiment: resample your observations with replacement, recompute the statistic, and repeat. The spread of those recomputed statistics is a stand-in for the spread you would see across new samples. A percentile interval reads off two quantiles of that spread. It is not magic. It assumes your sample is a reasonable picture of the population, and it fails when the sample is tiny or the statistic is fragile. The research module uses the same idea on model scores. Here you only need it for a mean, where the claim is easy to check.

## Math

A Bernoulli observation $x \in \{0,1\}$ has likelihood $\theta^x (1-\theta)^{1-x}$. For an i.i.d. sample the MLE is the mean:

$$
\hat\theta = \frac{1}{n}\sum_{i=1}^n x_i.
$$

A Gaussian sample has MLEs

$$
\hat\mu = \bar x, \qquad \hat\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \hat\mu)^2.
$$

Binary cross-entropy, clipped so a probability of 0 or 1 cannot produce an infinite log, is

$$
\mathrm{BCE}(y, p) = -\frac{1}{n}\sum_i \Big( y_i \log p_i + (1-y_i)\log(1-p_i) \Big).
$$

For a bootstrap confidence interval of the mean, draw $B$ resamples of size $n$ with replacement, compute each resample mean, sort them, and read the $\alpha/2$ and $1-\alpha/2$ percentiles by linear interpolation. Index $q$ in a sorted array of length $B$ sits at position $(B-1)q$.

## Resources

- [Seeing Theory: basic probability](https://seeing-theory.brown.edu/basic-probability/index.html) · reading · Kunin, Guo, and the Seeing Theory project · Visual definitions of probability, expectation, and sampling.
- [Probability and statistics](https://d2l.ai/chapter_preliminaries/probability.html) · reading · Dive into Deep Learning · Random variables and expectation in the notation deep-learning code uses.
- [Maximum likelihood](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/maximum-likelihood.html) · reading · Dive into Deep Learning · The MLE principle, including the Gaussian case.
- [Information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html) · reading · Dive into Deep Learning · Entropy and cross-entropy, which is why the loss has this name.
- [Think Stats, 3rd edition](https://greenteapress.com/wp/think-stats-3e/) · book · Allen Downey · A free book that treats resampling as a first method, not an appendix.
- [Mathematics for Machine Learning](https://mml-book.github.io/book/mml-book.pdf) · book · Deisenroth, Faisal, and Ong · The probability chapters, when you want the measure-theoretic caution removed but the definitions kept sharp.

## Exercise

Implement four functions.

- `bernoulli_mle(x)` returns the MLE of $\theta$ for a 1-D sample of 0s and 1s, as a Python float.
- `gaussian_mle(x)` returns `(mean, variance)` using the MLE variance, the one that divides by $n$.
- `binary_cross_entropy(y, p, eps=1e-12)` returns the mean BCE as a Python float. Clip `p` into `[eps, 1-eps]` before the log.
- `bootstrap_mean_ci(x, n_boot=1000, alpha=0.05, seed=0)` returns `(low, high)`. Resample with `np.random.default_rng(seed)`. Use the linear percentile described above. Raise `ValueError` if `x` is empty or not 1-D, or if `alpha` is not strictly between 0 and 1.

```python starter
import numpy as np


def bernoulli_mle(x):
    raise NotImplementedError


def gaussian_mle(x):
    raise NotImplementedError


def binary_cross_entropy(y, p, eps=1e-12):
    raise NotImplementedError


def bootstrap_mean_ci(x, n_boot=1000, alpha=0.05, seed=0):
    raise NotImplementedError
```

```python solution
import numpy as np


def bernoulli_mle(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x))


def gaussian_mle(x):
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    variance = float(np.mean((x - mean) ** 2))
    return mean, variance


def binary_cross_entropy(y, p, eps=1e-12):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def bootstrap_mean_ci(x, n_boot=1000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty 1-D sample")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        means[i] = np.mean(x[rng.integers(0, n, size=n)])
    means.sort()
    last = means.shape[0] - 1

    def percentile(q):
        if last == 0:
            return float(means[0])
        position = last * q
        lo = int(np.floor(position))
        hi = int(np.ceil(position))
        if lo == hi:
            return float(means[lo])
        weight = position - lo
        return float(means[lo] * (1.0 - weight) + means[hi] * weight)

    return percentile(alpha / 2.0), percentile(1.0 - alpha / 2.0)
```

```python tests
import numpy as np

assert abs(bernoulli_mle([1, 0, 1, 1]) - 0.75) < 1e-12
mu, var = gaussian_mle(np.array([1.0, 2.0, 3.0]))
assert abs(mu - 2.0) < 1e-12
assert abs(var - (2.0 / 3.0)) < 1e-12

y = np.array([1.0, 0.0])
p = np.array([0.5, 0.5])
assert abs(binary_cross_entropy(y, p) - np.log(2)) < 1e-12
assert np.isfinite(binary_cross_entropy(np.array([1.0]), np.array([0.0])))
assert np.isfinite(binary_cross_entropy(np.array([0.0]), np.array([1.0])))

low, high = bootstrap_mean_ci(np.array([5.0, 5.0, 5.0, 5.0]), n_boot=30, seed=0)
assert abs(low - 5.0) < 1e-12 and abs(high - 5.0) < 1e-12
sample = np.array([1.0, 2.0, 3.0, 4.0, 8.0])
low, high = bootstrap_mean_ci(sample, n_boot=400, alpha=0.1, seed=1)
assert low <= float(sample.mean()) <= high
assert bootstrap_mean_ci(sample, n_boot=400, alpha=0.1, seed=1) == (low, high)
try:
    bootstrap_mean_ci(sample, alpha=1.2)
    raise AssertionError("alpha must be validated")
except ValueError:
    pass
try:
    bootstrap_mean_ci(np.zeros((2, 2)))
    raise AssertionError("x must be 1-D")
except ValueError:
    pass
```

## Quiz

### q1
You flip a coin 10 times and see 7 heads. The Bernoulli MLE for the heads probability is

- [ ] $7/9$, because one degree of freedom was used.
- [x] $0.7$
- [ ] $0.5$, unless you have a prior.
- [ ] impossible to compute without the order of the flips.

> The likelihood depends on the count, not the order. The maximizer of $\theta^7(1-\theta)^3$ is $7/10$.

### q2
Why is binary cross-entropy the same objective as Bernoulli maximum likelihood?

- [ ] Cross-entropy squares the residual, and a Bernoulli variance equals $p(1-p)$.
- [x] BCE is the average negative log probability of the labels under predicted Bernoulli probabilities.
- [ ] They match only when every label is 1.
- [ ] MLE maximizes accuracy, and BCE is a smooth accuracy.

> Minimizing the negative log-likelihood is maximizing the likelihood. The average is a constant factor and does not move the minimizer.

### q3
A bootstrap percentile interval for the mean is built by

- [ ] adding and subtracting 1.96 sample standard deviations, always.
- [ ] dropping the largest and smallest observations once.
- [x] recomputing the mean on many resamples drawn with replacement, then reading two quantiles.
- [ ] fitting a Gaussian by gradient descent.

> The interval is a summary of a simulated sampling distribution. The normal formula is a different interval, available when the mean is roughly normal and you are willing to use that model.
