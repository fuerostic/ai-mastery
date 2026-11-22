---
id: macro-f1
module: research
title: Precision, recall, and macro-F1
minutes: 35
summary: Accuracy hides a rare class. Macro-F1 gives that class an equal vote.
idea: Decide which mistakes count before you celebrate a single number.
---

## Intuition

Accuracy is the fraction of labels you got right. It is the right summary when classes are balanced and every mistake costs the same. NLP evaluation often fails both assumptions. A sentiment dataset can be mostly neutral. A rare-event tagger can score 99% accuracy by predicting "none" and missing every event you built the dataset to find.

Precision, for one class, is the fraction of your predicted positives that were real. Recall is the fraction of the real positives you found. F1 is their harmonic mean, so a zero on either side is a zero overall. You cannot buy an F1 with perfect precision and no recall.

Macro-F1 computes that F1 once per class and then takes the unweighted mean. A class with 10 examples counts as much as a class with 10,000. Micro-F1 pools the counts first, so large classes dominate, and in a single-label multiclass problem it collapses toward accuracy. If you do not say which one you computed, the number is not interpretable. This lesson always macro-averages over a fixed label set $\{0,\ldots,C-1\}$, including classes that happen to be absent from the slice you scored. An absent class contributes F1 0 under the convention below. Some papers instead skip classes with no gold support. Say which one you did.

## Math

From a confusion matrix $C$ where rows are gold and columns are predicted,

$$
\mathrm{TP}_c = C_{cc}, \quad
\mathrm{FP}_c = \sum_i C_{ic} - \mathrm{TP}_c, \quad
\mathrm{FN}_c = \sum_j C_{cj} - \mathrm{TP}_c.
$$

$$
P_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c}, \qquad
R_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FN}_c}, \qquad
F_c = \frac{2 P_c R_c}{P_c + R_c}.
$$

If a denominator is 0, this lesson sets that precision or recall to 0. If $P_c + R_c = 0$, then $F_c = 0$. Macro-F1 is the unweighted mean of $F_c$ over $c = 0,\ldots,C-1$.

## Resources

- [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) · reading · scikit-learn · The definitions of precision, recall, F1, and the averaging modes, written next to the functions people actually call.
- [f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) · reading · scikit-learn · The `average='macro'` switch, and the zero-division policy you must set on purpose.
- [An Introduction to Statistical Learning](https://www.statlearning.com/) · book · James, Witten, Hastie, and Tibshirani · Classification metrics in the free textbook, before anyone mentions a leaderboard.
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) · course · Stanford · The NLP course where these metrics are attached to real sequence and classification tasks.

## Exercise

Implement a confusion matrix and macro-F1.

- `confusion_matrix(y_true, y_pred, n_classes)` returns an integer array shaped `(n_classes, n_classes)`, rows gold, columns predicted. Labels are integers in `0 .. n_classes-1`.
- `macro_f1(y_true, y_pred, n_classes)` returns the unweighted mean F1 as a Python float, using the zero-denominator convention above.

For `y_true = [0, 0, 1, 1]` and `y_pred = [0, 1, 1, 1]`, class 0 has F1 $2/3$ and class 1 has F1 $4/5$, so macro-F1 is $11/15$.

```python starter
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    raise NotImplementedError


def macro_f1(y_true, y_pred, n_classes):
    raise NotImplementedError
```

```python solution
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    matrix = np.zeros((int(n_classes), int(n_classes)), dtype=int)
    for gold, pred in zip(y_true, y_pred):
        matrix[gold, pred] += 1
    return matrix


def macro_f1(y_true, y_pred, n_classes):
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    scores = []
    for label in range(matrix.shape[0]):
        tp = matrix[label, label]
        fp = matrix[:, label].sum() - tp
        fn = matrix[label, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))
```

```python tests
import numpy as np

y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 1])
matrix = confusion_matrix(y_true, y_pred, 2)
assert np.array_equal(matrix, np.array([[1, 1], [0, 2]]))
assert abs(macro_f1(y_true, y_pred, 2) - (11.0 / 15.0)) < 1e-12

perfect = np.array([0, 1, 2, 2])
assert abs(macro_f1(perfect, perfect, 3) - 1.0) < 1e-12
# Class 2 has no support and no predictions: it contributes 0, so the macro score is 2/3.
binary = np.array([0, 1, 0, 1])
assert abs(macro_f1(binary, binary, 3) - (2.0 / 3.0)) < 1e-12
```

## Quiz

### q1
Macro-F1 averages

- [ ] one F1 computed from the pooled true positives of every class.
- [x] the per-class F1 scores, with each class weighted equally.
- [ ] accuracy, but only on the rarest class.
- [ ] precision minus recall.

> Equal class weights are the point. Micro-averaging pools counts first and lets large classes dominate.

### q2
A class with no gold examples and no predictions has, under this lesson's convention,

- [ ] F1 equal to 1, because there was nothing to get wrong.
- [x] F1 equal to 0, because both denominators are zero and the convention fills them with 0.
- [ ] an undefined F1, and macro-F1 is therefore undefined.
- [ ] F1 equal to the accuracy.

> This is a choice, and it punishes label sets that contain classes missing from the slice. Skipping empty classes is the other common choice. Report which one you used.

### q3
Precision 1 and recall 0 produce an F1 of

- [ ] 1
- [ ] 0.5
- [x] 0
- [ ] undefined, because the harmonic mean of those two numbers does not exist

> The harmonic mean is zero when either input is zero. Predicting nothing is not a high-precision success under F1.
