---
id: annotation-kappa
module: research
title: Annotation and Cohen's kappa
minutes: 35
summary: Two annotators can agree often and still be doing no better than the label frequencies.
idea: Report raw agreement and chance-corrected agreement. One number hides the marginals.
---

## Intuition

A label is a decision someone made, under a guideline, on a particular day. Before you train on those decisions or crown a model for matching them, you want to know whether a second person would have decided the same way. Raw agreement is the fraction of items where the two annotators match. It is easy to overread. If both annotators mark 95% of tweets "not hate", they will agree about 90% of the time by never using the rare label. That is not evidence the guideline is clear.

Cohen's kappa subtracts the agreement you would expect from the annotators' own marginal frequencies, assuming independence, and rescales what remains:

$$
\kappa = \frac{p_o - p_e}{1 - p_e}.
$$

One is perfect agreement. Zero is exactly chance. Negative means the annotators agree less than chance, which happens, and which a metric that bottoms out at zero would hide. Kappa is not a property of the data alone. It depends on the marginals. The same disagreement pattern can move $\kappa$ when one annotator becomes more balanced. This is the prevalence problem, and it is why you publish the confusion matrix between annotators along with kappa. The Artstein and Poesio survey is the reading that stops people from treating 0.8 as a law of nature. There is no universal threshold. There is a number, a matrix, and a guideline.

Disagreement is also information. Items where trained annotators split are often the items a model should not be punished for missing in a single-gold evaluation. That is an argument about the task, not a license to delete the hard examples quietly.

## Math

For $n$ items, $C$ classes, and two label arrays, let $M_{ab}$ be the number of items labeled $a$ by the first annotator and $b$ by the second.

$$
p_o = \frac{\mathrm{tr}(M)}{n}, \qquad
p_e = \sum_c \frac{M_{c\cdot}}{n} \frac{M_{\cdot c}}{n}.
$$

$M_{c\cdot}$ is a row sum and $M_{\cdot c}$ is a column sum. If $p_e = 1$, both annotators used a single class and agreed, and this lesson defines $\kappa = 1$. Otherwise use the fraction above.

A worked case: labels `[0, 0, 1, 1]` against `[0, 0, 1, 0]`. Then $p_o = 0.75$, the marginal products give $p_e = 0.5$, and $\kappa = 0.5$.

## Resources

- [Survey article: inter-coder agreement for computational linguistics](https://aclanthology.org/J08-4004/) · paper · Artstein and Poesio · The survey that explains kappa, its variants, and why a single cutoff is not a method.
- [cohen_kappa_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html) · reading · scikit-learn · The function you will eventually call, and the weighting options this lesson does not implement.
- [The Hitchhiker's Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/) · paper · Dror et al. · How agreement numbers and significance tests get misused in the same papers.
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) · course · Stanford · Data collection and evaluation as part of the NLP curriculum, not as an afterthought.

## Exercise

Implement `cohens_kappa(y1, y2, n_classes)`.

Both label arrays are 1-D integers in `0 .. n_classes-1` and the same length. Return a Python float. Use the definition above, including $\kappa = 1$ when $p_e = 1$.

```python starter
import numpy as np


def cohens_kappa(y1, y2, n_classes):
    raise NotImplementedError
```

```python solution
import numpy as np


def cohens_kappa(y1, y2, n_classes):
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    n = y1.shape[0]
    matrix = np.zeros((int(n_classes), int(n_classes)), dtype=float)
    for left, right in zip(y1, y2):
        matrix[left, right] += 1.0
    observed = np.trace(matrix) / n
    expected = float(np.sum(matrix.sum(axis=0) * matrix.sum(axis=1)) / (n * n))
    if expected == 1.0:
        return 1.0
    return float((observed - expected) / (1.0 - expected))
```

```python tests
import numpy as np

perfect = np.array([0, 1, 1, 0, 2])
assert abs(cohens_kappa(perfect, perfect, 3) - 1.0) < 1e-12
constant = np.zeros(5, dtype=int)
assert abs(cohens_kappa(constant, constant, 2) - 1.0) < 1e-12

y1 = np.array([0, 0, 1, 1])
y2 = np.array([0, 0, 1, 0])
assert abs(cohens_kappa(y1, y2, 2) - 0.5) < 1e-12
```

## Quiz

### q1
Raw agreement of 0.9 does not imply a clear guideline when

- [ ] kappa is also 0.9.
- [x] both annotators rarely use some labels, so chance agreement is already high.
- [ ] there are more than two classes.
- [ ] the dataset is public.

> Chance agreement is computed from the marginals. Skewed marginals make accidental agreement easy, and kappa is the correction for that.

### q2
$\kappa = 0$ means

- [ ] the annotators disagreed on every item.
- [x] the annotators agreed exactly as often as their marginal frequencies predict under independence.
- [ ] the guideline should be discarded automatically.
- [ ] macro-F1 against either annotator is zero.

> Zero is the chance baseline of this coefficient, not the floor of agreement. Perfect disagreement can be negative.

### q3
Why publish the annotator confusion matrix with kappa?

- [ ] Kappa is undefined without the matrix trace, so readers must recompute it.
- [x] Kappa reacts to the marginals. The matrix shows whether the remaining disagreement is one rare class or a general split.
- [ ] The matrix is a significance test.
- [ ] Cohen's kappa only supports two classes, and the matrix generalizes it.

> A single coefficient cannot tell you that all the trouble is "sarcasm versus not" rather than "positive versus negative". The matrix can.
