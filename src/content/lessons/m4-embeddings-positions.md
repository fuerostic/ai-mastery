---
id: embeddings-positions
module: transformers
title: Embeddings and sinusoidal positions
minutes: 35
summary: Tokens become vectors, and position has to be added because attention itself has none.
idea: A table lookup is the embedding. The sinusoid is a position the model did not have to learn.
---

## Intuition

An embedding is a learned vector for each token id. The forward pass does not multiply by a gigantic one-hot matrix in any implementation you should write. It indexes a table. Row 17 is the vector for token 17. That row is still the thing a gradient will move, and at training time it behaves like the corresponding column of a weight matrix. The lookup is a convenience, not a different model.

Attention, in the next lesson, is a weighted average of value vectors. The weights depend on content. They do not depend on order unless you put order somewhere. Permute the tokens, and a bare attention layer permutes its outputs the same way. Language is not like that. "Dog bites person" and "person bites dog" are different sentences. Positional information has to enter the residual stream before the content-only mixing begins.

The original transformer adds a fixed sinusoidal pattern. Even dimensions are sines and odd dimensions are cosines, with wavelengths that grow geometrically with the dimension index. Nearby positions get similar vectors. Far positions get vectors that oscillate at different speeds in the low dimensions and barely move in the high ones. Later models often learn the position table instead, or put the position inside the attention scores as rotary embeddings. The sinusoid is still the one you should be able to write from the paper, because every variant is a comment on it.

Adding the position vector to the token vector is a sum, not a concatenation, in this design. The residual stream stays width $d_{\text{model}}$. The two signals share the coordinates and the rest of the network has to disentangle them.

## Math

A token id sequence $i_1, \ldots, i_T$ and an embedding table $E \in \mathbb{R}^{V \times d}$ give

$$
X_{t} = E_{i_t} + P_{t}.
$$

Sinusoidal positions, with $d$ even and frequency index $k = 0, \ldots, d/2 - 1$:

$$
\begin{aligned}
P_{t, 2k} &= \sin\!\big(t / 10000^{2k/d}\big), \\
P_{t, 2k+1} &= \cos\!\big(t / 10000^{2k/d}\big).
\end{aligned}
$$

At position 0 every sine is 0 and every cosine is 1, so the vector is $(0, 1, 0, 1, \ldots)$. The wavelength of pair $k$ is $2\pi \cdot 10000^{2k/d}$.

## Resources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · reading · Jay Alammar · The picture of embeddings entering the encoder, before any attention math.
- [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) · reading · Jay Alammar · What a learned token vector is, in the model that made embeddings famous.
- [Attention is all you need](https://arxiv.org/abs/1706.03762) · paper · Vaswani et al. · Section 3.5 is the sinusoidal position formula this lesson implements.
- [Self-attention and positional encoding](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html) · reading · Dive into Deep Learning · Positions, with the same sinusoid and a learned alternative beside it.
- [Word embeddings tutorial](https://docs.pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html) · reading · PyTorch · The table-lookup view of an embedding module.

## Exercise

Implement the lookup, the sinusoid, and the sum.

- `embedding_lookup(ids, table)` returns `table[ids]`. `ids` is any integer array. Do not modify `table`.
- `sinusoidal_positions(seq_len, d_model)` returns a float array shaped `(seq_len, d_model)`. Require `d_model` even and raise `ValueError` otherwise. Use the formula above, with position index starting at 0.
- `add_positional(token_embeddings, positions)` returns their sum. Raise `ValueError` if the shapes differ.

```python starter
import numpy as np


def embedding_lookup(ids, table):
    raise NotImplementedError


def sinusoidal_positions(seq_len, d_model):
    raise NotImplementedError


def add_positional(token_embeddings, positions):
    raise NotImplementedError
```

```python solution
import numpy as np


def embedding_lookup(ids, table):
    ids = np.asarray(ids, dtype=int)
    table = np.asarray(table, dtype=float)
    return table[ids]


def sinusoidal_positions(seq_len, d_model):
    if d_model % 2 != 0:
        raise ValueError("d_model must be even")
    positions = np.arange(seq_len)[:, None]
    pair = np.arange(d_model // 2)[None, :]
    angles = positions / np.power(10000.0, (2.0 * pair) / d_model)
    encoded = np.zeros((seq_len, d_model), dtype=float)
    encoded[:, 0::2] = np.sin(angles)
    encoded[:, 1::2] = np.cos(angles)
    return encoded


def add_positional(token_embeddings, positions):
    token_embeddings = np.asarray(token_embeddings, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if token_embeddings.shape != positions.shape:
        raise ValueError("token embeddings and positions must share a shape")
    return token_embeddings + positions
```

```python tests
import numpy as np

table = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
looked = embedding_lookup(np.array([2, 0]), table)
assert np.allclose(looked, np.array([[5.0, 6.0], [1.0, 2.0]]))

pe = sinusoidal_positions(2, 4)
assert pe.shape == (2, 4)
assert np.allclose(pe[0], np.array([0.0, 1.0, 0.0, 1.0]), atol=1e-8)
assert np.allclose(pe[1, 0], np.sin(1.0), atol=1e-8)
assert np.allclose(pe[1, 1], np.cos(1.0), atol=1e-8)
assert np.allclose(pe[1, 2], np.sin(0.01), atol=1e-8)
assert np.allclose(pe[1, 3], np.cos(0.01), atol=1e-8)
try:
    sinusoidal_positions(3, 3)
    raise AssertionError("odd d_model should be rejected")
except ValueError:
    pass

tokens = np.ones((2, 4))
assert np.allclose(add_positional(tokens, pe), tokens + pe)
try:
    add_positional(tokens, pe[:1])
    raise AssertionError("shape mismatches should be rejected")
except ValueError:
    pass
```

## Quiz

### q1
Without a positional signal, a self-attention layer

- [ ] cannot have more than one head.
- [x] produces a permutation of its outputs when its inputs are permuted. Order is invisible.
- [ ] reduces to a bag-of-words average with uniform weights only.
- [ ] is unable to learn an embedding table.

> The attention weights are functions of content. Reorder the contents and the same weights follow them. The layer is permutation equivariant until position breaks that symmetry.

### q2
At position $t = 0$, the sinusoidal vector is

- [ ] the zero vector.
- [ ] $(1, 0, 1, 0, \ldots)$
- [x] $(0, 1, 0, 1, \ldots)$
- [ ] a learned vector, so there is no closed form.

> $\sin(0) = 0$ and $\cos(0) = 1$ on every frequency pair. Even coordinates are the sines.

### q3
This design adds position to the token embedding, rather than concatenating, so that

- [ ] the softmax has twice as many classes.
- [x] the residual stream stays width $d_{\text{model}}$.
- [ ] sinusoidal values are probabilities and must be mixed by addition.
- [ ] concatenation would make the sines and cosines linearly dependent.

> Addition is a choice of interface. The rest of the block was drawn for vectors of width $d$, and the position signal occupies the same coordinates.
