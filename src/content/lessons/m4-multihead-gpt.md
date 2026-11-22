---
id: multihead-gpt
module: transformers
title: Multi-head attention and a GPT block
minutes: 45
summary: Several smaller attentions, concatenated, then a position-wise MLP. That is the block.
idea: One head can only look for one kind of relationship. The block is a small stack of those lookups and a feed-forward net.
---

## Intuition

A single attention distribution has one way to average the past. A head that is busy copying the previous token cannot also gather a subject that agrees with a verb three positions back. Multi-head attention runs several attentions side by side in slices of the embedding. Each head has its own query, key, and value projections, usually by slicing one big projection rather than allocating separate tensors. The heads write vectors of width $d / h$. Concatenate them and multiply by an output matrix $W_O$ to get back to width $d$.

With one head and identity projections, this reduces to the attention you just wrote. That is the test you should pass before trusting a four-head version. With a causal mask, changing a future token must not change an earlier output. If it does, a head forgot the mask or the concatenation mixed the times.

A GPT block is this attention plus a small position-wise network, each wrapped in a residual connection and a layer norm. The attention mixes tokens. The MLP mixes channels, independently at each position, and that is where most of the parameters live: two matrices, up to $4d$ and back. This lesson's exercise stops at multi-head attention, which is the piece people mis-implement. The block formula is here so you can read a GPT-2 diagram and know which tensor is which. Building the full sampling loop is what nanoGPT and the Raschka chapters are for. Do not skip them. Do not pretend this page replaced them.

## Math

Let $X \in \mathbb{R}^{T \times d}$ and $d = h \cdot d_h$.

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V,
$$

each in $\mathbb{R}^{T \times d}$. Split the last axis into $h$ heads of width $d_h$, run causal or bidirectional attention per head, concatenate back to $\mathbb{R}^{T \times d}$, and project:

$$
\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W_O.
$$

A pre-norm GPT block, in the form used by later GPT implementations, is

$$
\begin{aligned}
X &\leftarrow X + \mathrm{MHA}(\mathrm{LayerNorm}(X)), \\
X &\leftarrow X + \mathrm{MLP}(\mathrm{LayerNorm}(X)).
\end{aligned}
$$

The original transformer paper used post-norm. When you read a diagram, check which side the norm is on before you compare residuals. $W_Q, W_K, W_V, W_O$ are all $(d, d)$ in this lesson's layout, which stores matrices on the right of row vectors.

## Resources

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) · reading · Jay Alammar · The decoder-only stack, one block at a time.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · reading · Jay Alammar · Multi-head attention drawn before the GPT specialization.
- [Attention is all you need](https://arxiv.org/abs/1706.03762) · paper · Vaswani et al. · Section 3.2.2 is multi-head attention.
- [nanoGPT](https://github.com/karpathy/nanoGPT) · code · Andrej Karpathy · A small GPT you can read in one sitting after this lesson.
- [LLMs from scratch, chapter 3](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03) · code · Sebastian Raschka · Coding attention mechanisms, including the multi-head module.
- [LLMs from scratch, chapter 4](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04) · code · Sebastian Raschka · Assembling those pieces into a GPT.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) · code · Harvard NLP · The encoder-decoder original, when you want the paper and the code on the same page.
- [Language models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) · paper · Radford et al. · The GPT-2 technical report. Short, and specific about the decoder-only block.

## Exercise

Implement `multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=False)`.

`X` is `(T, d)`. The four weight matrices are `(d, d)`. `d` must be divisible by `n_heads`. Project, split into heads of width `d // n_heads`, run the scaled dot-product attention from the previous lesson in each head, concatenate in head order, and multiply by `Wo`. Return the `(T, d)` output.

Use the same causal convention as `attention`: a strictly upper-triangular mask of `-1e9` inside each head. Raise `ValueError` when `d` is not divisible by `n_heads`.

You can call a function named `attention` if you define it in this file. The tests only call `multi_head_attention`.

```python starter
import numpy as np


def attention(Q, K, V, causal=False):
    raise NotImplementedError


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=False):
    raise NotImplementedError
```

```python solution
import numpy as np


def attention(Q, K, V, causal=False):
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    scores = (Q @ K.T) / np.sqrt(Q.shape[-1])
    if causal:
        mask = np.triu(np.ones((Q.shape[0], K.shape[0]), dtype=bool), k=1)
        scores = scores.copy()
        scores[mask] = -1e9
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return weights @ V, weights


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=False):
    X = np.asarray(X, dtype=float)
    time, width = X.shape
    if width % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    head_dim = width // int(n_heads)

    def split(projected):
        return projected.reshape(time, int(n_heads), head_dim).transpose(1, 0, 2)

    queries, keys, values = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    contexts = []
    for head in range(int(n_heads)):
        context, _weights = attention(queries[head], keys[head], values[head], causal=causal)
        contexts.append(context)
    concatenated = np.stack(contexts, axis=1).reshape(time, width)
    return concatenated @ Wo
```

```python tests
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(4, 4))
eye = np.eye(4)
single = multi_head_attention(X, eye, eye, eye, eye, n_heads=1, causal=False)
scores = (X @ X.T) / np.sqrt(4)
scores = scores - scores.max(axis=1, keepdims=True)
weights = np.exp(scores)
weights = weights / weights.sum(axis=1, keepdims=True)
assert np.allclose(single, weights @ X, atol=1e-6)

causal = multi_head_attention(X, eye, eye, eye, eye, n_heads=2, causal=True)
changed = X.copy()
changed[3] += 10.0
causal_changed = multi_head_attention(changed, eye, eye, eye, eye, n_heads=2, causal=True)
assert np.allclose(causal[:3], causal_changed[:3], atol=1e-6)
assert not np.allclose(causal[3], causal_changed[3])

Wq = rng.normal(size=(4, 4))
Wk = rng.normal(size=(4, 4))
Wv = rng.normal(size=(4, 4))
Wo = rng.normal(size=(4, 4))
out = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=2, causal=True)
assert out.shape == X.shape
try:
    multi_head_attention(X, eye, eye, eye, eye, n_heads=3)
    raise AssertionError("n_heads must divide d_model")
except ValueError:
    pass
```

## Quiz

### q1
With `n_heads = 1` and all four projections equal to the identity, multi-head attention equals

- [ ] the sum of the value vectors, unnormalized.
- [x] scaled dot-product attention of $X$ with itself.
- [ ] a convolution of width 1.
- [ ] the positional sinusoid.

> One head sees the full width. Identity maps leave $Q = K = V = X$, and $W_O$ does not move the result.

### q2
A causal multi-head block is implemented correctly only if

- [ ] every head uses a different mask.
- [x] editing a future token leaves earlier outputs unchanged.
- [ ] the output projection is diagonal.
- [ ] heads are summed rather than concatenated.

> Each head has to honor the same time mask. Concatenation and $W_O$ mix channels, not times, so they cannot put future information into an earlier row unless a head already did.

### q3
In a GPT block the position-wise MLP

- [ ] mixes tokens and leaves channels untouched.
- [x] mixes channels at each position separately. Token mixing already happened in the attention.
- [ ] replaces the need for positional encodings.
- [ ] is applied only to the first token.

> Attention is the token mixer. The MLP is the channel mixer, and it is where the parameter count usually concentrates.
