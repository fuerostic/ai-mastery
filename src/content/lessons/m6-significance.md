---
id: significance
module: research
title: Paired bootstrap intervals and permutation tests
minutes: 40
summary: A test-set score is a random variable. Resampling is how you look at its noise.
idea: Report an interval or a comparison, not a lone point, when two systems are close.
---

## Intuition

You evaluated two taggers on the same 800 sentences. One scores 0.4 points of F1 higher. That gap might be the system, or it might be those 800 sentences. A different test draw would move both numbers. The bootstrap from the probability lesson resamples the test set with replacement and recomputes the statistic. When the two systems were scored on the same items, resample items, not the two score lists separately. The pair stays together. The interval you want is an interval on the difference.

A permutation test asks a sharper question: if the two systems were exchangeable, how often would a gap at least this large appear? Pool the observations, shuffle the labels, recompute the difference, and count. This lesson uses the add-one version $(count + 1) / (n_{\mathrm{perm}} + 1)$ so a p-value is never reported as zero. A zero would claim an impossibility you did not prove. Add-one also means the smallest p-value you can see with 1000 shuffles is about 0.001.

Neither procedure fixes a bad test set. If the test set is contaminated, or the metric is the wrong one, a tight interval just makes you confidently wrong. The Dror et al. paper linked below is the field's own warning label: pick the test that matches the dependence in your data, and do not treat a p-value as a substitute for an effect size. Report the interval and the difference. The p-value is the third number, not the only one.

## Math

For paired scores $a_i, b_i$, the statistic is $\bar a - \bar b$. A paired bootstrap draws indexes with replacement, recomputes the difference, and reads the same linear percentiles as the earlier mean interval.

An unpaired two-sided permutation p-value for the absolute difference of means, with $n_{\mathrm{perm}}$ shuffles of the pooled sample, is

$$
p = \frac{1 + \#\{\,|\bar a^\ast - \bar b^\ast| \ge |\bar a - \bar b|\,\}}{1 + n_{\mathrm{perm}}}.
$$

Use a fixed random generator so the same seed reproduces the same p-value. The comparison should tolerate a tiny numerical tolerance, on the order of $10^{-12}$, so a difference that is the observed one does not fall on the wrong side of `>=` because of rounding.

## Resources

- [The Hitchhiker's Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/) · paper · Dror, Baumer, Shlomov, and Reichart · Which test matches which experimental design, and the mistakes the field keeps repeating.
- [Think Stats, 3rd edition](https://greenteapress.com/wp/think-stats-3e/) · book · Allen Downey · Resampling taught as a primary tool, with code-level definitions of the procedures.
- [An Introduction to Statistical Learning](https://www.statlearning.com/) · book · James, Witten, Hastie, and Tibshirani · The resampling chapter in the free textbook: bootstrap and cross-validation, with the bias warnings attached.
- [Seeing Theory: basic probability](https://seeing-theory.brown.edu/basic-probability/index.html) · reading · Seeing Theory · A visual reminder of what a sampling distribution is, before the test-set version.

## Exercise

Implement a paired bootstrap interval and an unpaired permutation p-value.

- `paired_bootstrap_diff_ci(a, b, n_boot=1000, alpha=0.05, seed=0)` takes two 1-D arrays of equal length. Resample shared indexes with replacement. Return `(low, high)` for the mean of `a - b`, using linear interpolation on the sorted bootstrap differences at $\alpha/2$ and $1-\alpha/2$. Raise `ValueError` on a shape mismatch or an empty input.
- `permutation_pvalue(a, b, n_perm=1000, seed=0)` returns the add-one two-sided p-value for the absolute difference of means. `a` and `b` are independent samples, not necessarily the same length.

If `a = b + 1` elementwise, every resample has difference 1, so the interval is `(1, 1)`.

```python starter
import numpy as np


def paired_bootstrap_diff_ci(a, b, n_boot=1000, alpha=0.05, seed=0):
    raise NotImplementedError


def permutation_pvalue(a, b, n_perm=1000, seed=0):
    raise NotImplementedError
```

```python solution
import numpy as np


def _percentile(sorted_values, q):
    last = len(sorted_values) - 1
    if last <= 0:
        return float(sorted_values[0])
    position = last * q
    lo = int(np.floor(position))
    hi = int(np.ceil(position))
    if lo == hi:
        return float(sorted_values[lo])
    weight = position - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def paired_bootstrap_diff_ci(a, b, n_boot=1000, alpha=0.05, seed=0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be non-empty 1-D arrays of equal length")
    rng = np.random.default_rng(seed)
    diffs = np.empty(int(n_boot), dtype=float)
    n = a.shape[0]
    for i in range(int(n_boot)):
        index = rng.integers(0, n, size=n)
        diffs[i] = np.mean(a[index] - b[index])
    diffs.sort()
    return _percentile(diffs, alpha / 2.0), _percentile(diffs, 1.0 - alpha / 2.0)


def permutation_pvalue(a, b, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = abs(float(a.mean() - b.mean()))
    pooled = np.concatenate([a, b])
    n_a = a.shape[0]
    count = 0
    for _ in range(int(n_perm)):
        rng.shuffle(pooled)
        diff = abs(float(pooled[:n_a].mean() - pooled[n_a:].mean()))
        if diff >= observed - 1e-12:
            count += 1
    return float((count + 1) / (int(n_perm) + 1))
```

```python tests
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([0.0, 1.0, 2.0])
low, high = paired_bootstrap_diff_ci(a, b, n_boot=40, seed=0)
assert abs(low - 1.0) < 1e-9 and abs(high - 1.0) < 1e-9
again = paired_bootstrap_diff_ci(np.array([1.0, 3.0, 5.0, 7.0]), np.array([0.0, 1.0, 2.0, 4.0]), n_boot=200, seed=2)
assert again[0] <= 2.25 <= again[1]
assert paired_bootstrap_diff_ci(np.array([1.0, 3.0, 5.0, 7.0]), np.array([0.0, 1.0, 2.0, 4.0]), n_boot=200, seed=2) == again
try:
    paired_bootstrap_diff_ci(a, b[:2])
    raise AssertionError("unequal pairs should be rejected")
except ValueError:
    pass

identical = permutation_pvalue(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), n_perm=50, seed=0)
assert abs(identical - 1.0) < 1e-12
separated = permutation_pvalue(np.zeros(6), np.full(6, 5.0), n_perm=300, seed=1)
assert separated < 0.05
```

## Quiz

### q1
Scores from two systems on the same test items should be resampled

- [ ] as two independent lists, so the systems do not influence each other.
- [x] by shared indexes, so each draw compares the systems on the same resampled items.
- [ ] only if the scores are F1. Accuracy does not need pairing.
- [ ] by shuffling tokens inside each sentence.

> The dependence is the point of a paired test. Independent resampling invents test sets where system A saw item 3 and system B did not.

### q2
The add-one permutation p-value

- [ ] is always smaller than $1/n_{\mathrm{perm}}$, which is what makes it significant.
- [x] never reports 0, and equals 1 when every shuffle matches the observed gap, including the case of a zero gap.
- [ ] requires the two samples to have the same length.
- [ ] is a confidence interval in disguise.

> Adding one success and one trial is a hedge against claiming impossibility. A zero observed difference is matched by every shuffle, so the p-value is 1.

### q3
A narrow interval around a contaminated test set means

- [ ] the contamination cancelled out.
- [ ] the result is significant and therefore trustworthy.
- [x] the score is stable on that particular test file, which may still be the wrong file.
- [ ] you should switch from macro-F1 to accuracy.

> Resampling the test set estimates noise from drawing items out of that set. It does not audit where the set came from.
