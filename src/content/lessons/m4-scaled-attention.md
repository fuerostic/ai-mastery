---
id: scaled-attention
module: transformers
title: Scaled dot-product attention and causal masks
minutes: 40
summary: A weighted average of values, with weights from queries and keys, and a mask that hides the future.
idea: Attention is a soft lookup. The scale keeps the lookup from becoming a hard max.
---

## Intuition

A query asks. A key advertises. A value holds the content that will actually be mixed. For every query you take a dot product with every key, turn those scores into a probability distribution with softmax, and return the weighted average of the values. That is the whole mechanism. The name "attention" is the softmax. The algebra is a lookup that can land between rows instead of on one of them.

Dot products grow with the dimension of the vectors. If each coordinate of $q$ and $k$ is a modest random number, $q^\top k$ has variance proportional to $d_k$. Softmax of a large-magnitude vector is nearly one-hot, and its gradient is nearly zero. Dividing by $\sqrt{d_k}$ puts the scores back on a scale where the softmax still has a slope. The scale is not a decoration. Take it out and a randomly initialized attention layer is already saturated.

A causal language model is not allowed to look at future tokens while predicting the present one. The mask sets those scores to a large negative number before the softmax, so their weights become zero. Position 0 then attends only to itself. Position $t$ attends to positions $0$ through $t$. The same function without the mask is bidirectional, which is what an encoder wants and what a decoder must not do during training.

This lesson uses a single sequence, with $Q$, $K$, and $V$ shaped `(T, d)`. Batches and heads are the next lesson. Get the weights to sum to one, and get the future weights to vanish, before you stack anything.

## Math

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V.
$$

For a causal mask, with $M_{ij} = \mathbf{1}[j > i]$,

$$
S = \frac{QK^\top}{\sqrt{d_k}}, \qquad S_{ij} \leftarrow -10^{9} \text{ when } M_{ij} = 1,
$$

and the softmax is then row-wise. Subtract the row max before exponentiating, as in the softmax lesson. A row that was entirely masked still needs a finite softmax. In a causal mask the diagonal is always visible, so no row is empty.

## Resources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · reading · Jay Alammar · Queries, keys, and values drawn as the operation, not only as letters.
- [Attention is all you need](https://arxiv.org/abs/1706.03762) · paper · Vaswani et al. · Section 3.2.1 is scaled dot-product attention, including the reason for the $\sqrt{d_k}$.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) · code · Harvard NLP · A line-by-line implementation of the paper. Read the attention function after you have written yours.
- [Queries, keys, and values](https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html) · reading · Dive into Deep Learning · The lookup interpretation, with the averaging made explicit.
- [Attention and augmented recurrent networks](https://distill.pub/2016/augmented-rnns/) · reading · Olah and Carter · The visual essay that made attention weights something you look at.
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) · reading · Lilian Weng · A survey of the variants, once the scaled-dot-product case is boring.

## Exercise

Implement `attention(Q, K, V, causal=False)`.

`Q`, `K`, and `V` are 2-D arrays. `Q` and `K` share a last dimension `d_k`. `V` has the same number of rows as `K`. Return `(output, weights)` where `weights` is `(T_q, T_k)` and `output = weights @ V`.

Divide scores by $\sqrt{d_k}$. If `causal` is true, set strictly upper-triangular scores to `-1e9` before the softmax. The softmax is row-wise and numerically stable. Raise `ValueError` if any input is not 2-D.

```python starter
import numpy as np


def attention(Q, K, V, causal=False):
    raise NotImplementedError
```

```python solution
import numpy as np


def attention(Q, K, V, causal=False):
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K, and V must be 2-D")
    scores = (Q @ K.T) / np.sqrt(Q.shape[-1])
    if causal:
        mask = np.triu(np.ones((Q.shape[0], K.shape[0]), dtype=bool), k=1)
        scores = scores.copy()
        scores[mask] = -1e9
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return weights @ V, weights
```

```python tests
import numpy as np

Q = np.zeros((3, 2))
K = np.zeros((3, 2))
V = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 3.0]])
out, weights = attention(Q, K, V, causal=False)
assert np.allclose(weights, np.full((3, 3), 1.0 / 3.0))
assert np.allclose(out, np.repeat(V.mean(axis=0, keepdims=True), 3, axis=0))

out_c, weights_c = attention(Q, K, V, causal=True)
assert np.allclose(out_c[0], V[0])
assert np.allclose(weights_c[1, :2], np.array([0.5, 0.5]), atol=1e-6)
assert weights_c[1, 2] < 1e-6
assert weights_c[0, 1] < 1e-6
assert np.allclose(weights_c.sum(axis=1), np.ones(3))

try:
    attention(np.zeros(3), K, V)
    raise AssertionError("1-D inputs should be rejected")
except ValueError:
    pass
```

## Quiz

### q1
The factor $1/\sqrt{d_k}$ is there so that

- [ ] the attention weights sum to $d_k$.
- [x] random query-key dots do not grow with dimension and saturate the softmax.
- [ ] causal masks remain triangular after the softmax.
- [ ] values are forced onto the unit sphere.

> The dot product of two independent random vectors has variance proportional to the dimension. Softmax of a huge vector is a near one-hot with a near-zero gradient.

### q2
Under a causal mask, the first position's attention weights are

- [ ] uniform over the whole sequence.
- [ ] zero everywhere, because there is no past.
- [x] 1 on itself and 0 on every later position.
- [ ] undefined, because the row is fully masked.

> The diagonal is visible. Every future score is pushed to a large negative number, so the softmax puts all of the mass on the only legal key.

### q3
`output = weights @ V` means each output row is

- [ ] the key with the largest score.
- [x] a convex combination of the value rows, with coefficients from that query's softmax.
- [ ] the sum of the queries.
- [ ] $V$ multiplied by $\sqrt{d_k}$.

> Softmax weights are non-negative and sum to 1. The output is an average of values, soft when the scores are close and nearly a selection when one score dominates.
