---
id: baselines-contamination
module: research
title: Baselines, prompting, and contamination
minutes: 40
summary: A leaderboard number needs a dumb comparison and a reason to believe the test set was unseen.
idea: If you cannot beat the majority label, the model is not the story. If the test set was in pretraining, the story is false.
---

## Intuition

A result without a baseline is a mood. The majority-class predictor is the smallest baseline that still respects the label distribution: always guess the most common training label, and break ties toward the smallest label id so the choice is reproducible. On a skewed task its accuracy can look impressive. Your model has to beat that number by enough to matter, not merely to differ from it in the third digit. The lift is accuracy minus majority accuracy. Negative lift means the model is worse than a constant.

Prompting and fine-tuning answer different questions. A prompted frozen model asks whether the behavior is already in the pretrained distribution and can be elicited by context. Fine-tuning asks whether a few thousand labeled examples can move the distribution toward the label you care about. Prompting wins when labels are scarce and the task looks like something the pretraining data already contained. Fine-tuning wins when the label is specific, the metric is the one you trained, and you have enough examples that the update is not memorizing five of them. FLAN-style instruction tuning is a third point on that line: fine-tune on many tasks so that prompting works better on the next one. Report which regime you are in. "We used an LLM" is not a regime.

Contamination is the failure mode that makes both numbers theatrical. If a benchmark was in the pretraining corpus, a prompted model can recite it. Exact normalized string match is a weak detector. It misses paraphrases, and the papers below document how common that gap is. It is still the check you can run this afternoon. A high exact-match rate is already a reason to distrust the evaluation. A low rate is not a proof of cleanliness. Normalize by stripping and lowercasing before you compare, so `"Hello"` and `" hello "` count as the same string. Empty test lists contribute a rate of 0.

## Math

Let $m$ be the smallest label among those with maximal count in `y_true`. Majority accuracy is the fraction of labels equal to $m$.

$$
\mathrm{lift} = \mathrm{accuracy}(y, \hat y) - \mathrm{accuracy}(y, m).
$$

Exact contamination, after $f(s) = \mathrm{lower}(\mathrm{strip}(s))$, is

$$
\frac{|\{ t \in \mathrm{test} : f(t) \in f(\mathrm{train}) \}|}{|\mathrm{test}|}.
$$

The train side is a set. Duplicate train strings do not count twice. Duplicate test strings do, because each test item is an evaluation item.

## Resources

- [Time travel in LLMs: tracing data contamination](https://arxiv.org/abs/2308.08493) · paper · Golchin and Surdeanu · A practical look at how evaluation sets leak into training corpora.
- [Rethinking benchmark and contamination for language models with rephrased samples](https://arxiv.org/abs/2311.04850) · paper · Yang et al. · Why exact string match underestimates contamination once the test item has been paraphrased.
- [Benchmark data contamination of large language models: a survey](https://arxiv.org/abs/2406.04244) · paper · Xu et al. · A map of the definitions. Read it before inventing a private one.
- [Finetuned language models are zero-shot learners](https://arxiv.org/abs/2109.01652) · paper · Wei et al. · Instruction fine-tuning as the bridge between a prompted model and a task-specific one.
- [Prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) · reading · Lilian Weng · What prompting is actually varying, once you stop treating the prompt as a spell.
- [An Introduction to Statistical Learning](https://www.statlearning.com/) · book · James, Witten, Hastie, and Tibshirani · Baselines and test error, in a book that does not assume the model is a language model.

## Exercise

Implement the majority baseline, its lift, and an exact contamination rate.

- `majority_label(y)` returns the smallest label id among the modes of a 1-D label array.
- `majority_accuracy(y)` returns the accuracy of always predicting that label, as a Python float.
- `accuracy(y_true, y_pred)` returns the fraction of equal entries.
- `lift_over_majority(y_true, y_pred)` returns accuracy minus majority accuracy.
- `exact_contamination_rate(train_texts, test_texts)` compares stripped, lowercased strings. An empty test list returns `0.0`.

```python starter
import numpy as np


def majority_label(y):
    raise NotImplementedError


def majority_accuracy(y):
    raise NotImplementedError


def accuracy(y_true, y_pred):
    raise NotImplementedError


def lift_over_majority(y_true, y_pred):
    raise NotImplementedError


def exact_contamination_rate(train_texts, test_texts):
    raise NotImplementedError
```

```python solution
import numpy as np


def majority_label(y):
    y = np.asarray(y)
    values, counts = np.unique(y, return_counts=True)
    best = counts.max()
    return values[counts == best].min()


def majority_accuracy(y):
    y = np.asarray(y)
    label = majority_label(y)
    return float(np.mean(y == label))


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def lift_over_majority(y_true, y_pred):
    return accuracy(y_true, y_pred) - majority_accuracy(y_true)


def exact_contamination_rate(train_texts, test_texts):
    if len(test_texts) == 0:
        return 0.0
    train = {text.strip().lower() for text in train_texts}
    hits = sum(1 for text in test_texts if text.strip().lower() in train)
    return float(hits / len(test_texts))
```

```python tests
import numpy as np

labels = np.array([1, 1, 0, 2, 1])
assert majority_label(labels) == 1
assert abs(majority_accuracy(labels) - 0.6) < 1e-12
tied = np.array([0, 1, 0, 1])
assert majority_label(tied) == 0
assert abs(majority_accuracy(tied) - 0.5) < 1e-12

pred = np.array([1, 1, 1, 1, 1])
assert abs(accuracy(labels, pred) - 0.6) < 1e-12
assert abs(lift_over_majority(labels, pred)) < 1e-12
better = np.array([1, 1, 0, 2, 0])
assert lift_over_majority(labels, better) > 0

rate = exact_contamination_rate(["Hello World", "Other"], [" hello world ", "no", "Hello World"])
assert abs(rate - (2.0 / 3.0)) < 1e-12
assert exact_contamination_rate(["a"], []) == 0.0
```

## Quiz

### q1
On labels `[0, 0, 0, 1]`, the majority baseline's accuracy is

- [ ] 0.5
- [x] 0.75
- [ ] 1, because the majority class is predictable from the index
- [ ] undefined without a model

> Three of the four labels are 0. Always predicting 0 scores 0.75. A model at 0.80 has lifted the baseline by 0.05, which may or may not be a result.

### q2
Prompting a frozen model and fine-tuning it are different experiments because

- [ ] prompting cannot use a test set.
- [x] prompting asks what the current parameters already do, while fine-tuning spends labeled examples to move those parameters.
- [ ] fine-tuning does not require a metric.
- [ ] a prompted model is immune to contamination.

> Both can be contaminated. The difference is whether the measured behavior was already in the model or was installed by the labeled update.

### q3
An exact-match contamination rate of 0 means

- [ ] the model cannot have seen the test distribution.
- [x] no stripped, lowercased test string appeared verbatim in the train list you checked. Paraphrases are invisible to this test.
- [ ] the leaderboard number is therefore causal.
- [ ] you should skip the majority baseline.

> Exact match is a screen, not an acquittal. The rephrasing papers exist because this screen misses the interesting leaks.
