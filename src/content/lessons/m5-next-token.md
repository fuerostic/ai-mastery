---
id: next-token
module: training
title: The pretraining objective
minutes: 35
summary: Predict the next token. The loss is cross-entropy, and padding must not count.
idea: A language model is a classifier whose classes are the vocabulary and whose label is the following token.
---

## Intuition

Pretraining, for a decoder-only model, is a loop over documents. Read a prefix. Predict a distribution over the next token. Compare it to the token that was actually there. Move the parameters to make that token a little more likely. There is no separate "understanding" loss hiding behind this. The capability people notice later is whatever this objective found useful for guessing the next token on a huge corpus.

The tensor layout is worth fixing in your head. A batch of token ids has shape `(N, T)`. The model eats the prefix `ids[:, :-1]` and is scored against `ids[:, 1:]`. Logits therefore have shape `(N, T-1, V)`. Many implementations instead run the model on the full sequence and shift inside the loss. Either convention is fine if the alignment is the one you think it is. This lesson makes the shift explicit and then scores already-aligned logits.

Padding is the other trap. A batch is rectangular, so short documents are filled with a pad id. Those positions are not tokens the model should be ashamed of missing. The loss ignores them. If you forget, the model spends capacity predicting padding, and a low loss stops meaning what you think it means.

The number you log is the mean negative log-probability of the non-ignored tokens. Perplexity is the exponential of that mean, in nats if you used natural logs. A uniform distribution over $V$ classes has loss $\log V$. That is the chance baseline, and a model that cannot beat it has not started.

## Math

Given aligned logits $Z \in \mathbb{R}^{N \times T \times V}$ and targets $y \in \{0,\ldots,V-1\}^{N \times T}$,

$$
L = -\frac{\sum_{n,t} \mathbf{1}[y_{nt} \neq \mathrm{pad}] \log \mathrm{softmax}(Z_{nt})_{y_{nt}}}{\sum_{n,t} \mathbf{1}[y_{nt} \neq \mathrm{pad}]}.
$$

The log-softmax of a row is $z_j - \max z - \log \sum_k \exp(z_k - \max z)$. If every position is padding, define the loss to be 0 rather than dividing by zero.

The shift that builds the aligned pair from a token batch is

$$
X = \mathrm{ids}_{:,\,:T-1}, \qquad y = \mathrm{ids}_{:,\,1:}.
$$

## Resources

- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) · video · Andrej Karpathy · The GPT lecture derives this loss and then trains it.
- [nanoGPT](https://github.com/karpathy/nanoGPT) · code · Andrej Karpathy · A training loop whose loss is this function and not much else.
- [LLMs from scratch, chapter 2](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02) · code · Sebastian Raschka · From text to input-target pairs.
- [LLMs from scratch, chapter 5](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05) · code · Sebastian Raschka · Pretraining on unlabeled data, including the loss you are implementing.
- [Language models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) · paper · Radford et al. · GPT-2, and a clear statement of what next-token prediction was asked to do.
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) · course · Hugging Face · The map of the ecosystem this objective sits under.

## Exercise

Implement the shift and the masked mean loss.

- `shift_batch(token_ids)` takes `(N, T)` with `T >= 2` and returns `(inputs, targets)` equal to `ids[:, :-1]` and `ids[:, 1:]`. Raise `ValueError` otherwise.
- `next_token_nll(logits, targets, ignore_index=None)` takes logits `(N, T, V)` and integer targets `(N, T)`. Return the mean negative log-probability over positions whose target is not `ignore_index`. If `ignore_index` is `None`, every position counts. If every position is ignored, return `0.0`.

```python starter
import numpy as np


def shift_batch(token_ids):
    raise NotImplementedError


def next_token_nll(logits, targets, ignore_index=None):
    raise NotImplementedError
```

```python solution
import numpy as np


def shift_batch(token_ids):
    ids = np.asarray(token_ids)
    if ids.ndim != 2 or ids.shape[1] < 2:
        raise ValueError("token_ids must have shape (batch, time) with time >= 2")
    return ids[:, :-1], ids[:, 1:]


def next_token_nll(logits, targets, ignore_index=None):
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    if ignore_index is None:
        mask = np.ones(flat_targets.shape[0], dtype=bool)
    else:
        mask = flat_targets != int(ignore_index)
    if not np.any(mask):
        return 0.0
    flat_logits = flat_logits[mask]
    flat_targets = flat_targets[mask]
    shifted = flat_logits - np.max(flat_logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    nll = -log_probs[np.arange(flat_targets.shape[0]), flat_targets]
    return float(np.mean(nll))
```

```python tests
import numpy as np

ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
inputs, targets = shift_batch(ids)
assert np.array_equal(inputs, np.array([[1, 2, 3], [5, 6, 7]]))
assert np.array_equal(targets, np.array([[2, 3, 4], [6, 7, 8]]))
try:
    shift_batch(np.array([[1]]))
    raise AssertionError("time length 1 should be rejected")
except ValueError:
    pass

uniform = np.zeros((2, 3, 4))
labels = np.array([[0, 1, 2], [3, 0, 1]])
assert abs(next_token_nll(uniform, labels) - np.log(4)) < 1e-8

padded = labels.copy()
padded[:, 1:] = -100
masked = next_token_nll(uniform, padded, ignore_index=-100)
assert abs(masked - np.log(4)) < 1e-8
assert next_token_nll(uniform, np.full_like(labels, -100), ignore_index=-100) == 0.0
```

## Quiz

### q1
For a sequence of token ids `[a, b, c, d]`, the pretraining targets aligned with the prefixes are

- [ ] `[a, b, c, d]`
- [x] `[b, c, d]`
- [ ] `[d, c, b, a]`
- [ ] a single vector, the mean of the embeddings

> Each position is scored on the token that follows it. The last token has no target inside this sequence, and the first token is not a target of an empty prefix.

### q2
A uniform distribution over a vocabulary of size $V$ has next-token loss

- [ ] $1/V$
- [ ] $0$, because every class is equally wrong.
- [x] $\log V$
- [ ] $V$

> Each token has probability $1/V$, so the negative log is $\log V$. Beating that number is the first sign of life.

### q3
Padding positions are ignored in the loss because

- [ ] their embeddings are the zero vector and have no gradient.
- [x] they are an artifact of rectangular batches, not tokens the model was supposed to predict.
- [ ] softmax is undefined on a pad id.
- [ ] the causal mask already removes them.

> The mask in attention and the ignore index in the loss do different jobs. A pad id can still sit in the target tensor, and the loss has to be told not to score it.
