---
id: eigen-pca-svd
module: math
title: Eigenvalues, PCA, and the SVD
minutes: 45
summary: Special directions that a matrix only stretches, and the coordinates PCA keeps.
idea: PCA is the covariance matrix telling you its longest axes.
---

## Intuition

Some directions are special for a linear map. Most vectors get rotated and stretched. An eigenvector only gets stretched: $Av$ lands on the same line as $v$, and the eigenvalue $\lambda$ is the stretch factor. Negative $\lambda$ flips the arrow. A zero eigenvalue flattens that direction away.

A cloud of data has its own stretch. Center the points, then ask which directions have the most variance. The sample covariance matrix answers that, and because covariance is symmetric its eigenvectors are perpendicular. Principal component analysis keeps the directions with the largest eigenvalues and throws the short ones out. You have changed coordinates, not invented new information. Reconstruction from all of the components gets the centered data back exactly. Reconstruction from the first few is the best low-rank linear summary in the least-squares sense.

The singular value decomposition says the same thing about any rectangular matrix, not only covariance. Every linear map is a rotation, an axis-aligned stretch, and another rotation: $X = U\Sigma V^\top$. The right singular vectors, the rows of $V^\top$, are the principal axes of $X^\top X$. For centered data they match the PCA axes, up to a sign that we will fix on purpose so the answer is unique.

Signs of eigenvectors are arbitrary. A vector and its negative span the same line. This lesson's convention: flip each component so that its largest-magnitude entry is positive. If two entries tie, `argmax` keeps the smallest index, and that is the tie-break.

## Math

An eigenpair of a square matrix satisfies

$$
Av = \lambda v, \qquad v \neq 0.
$$

Center an $n \times d$ data matrix $X$ by its column mean $\mu$. This lesson uses the unbiased sample covariance

$$
C = \frac{1}{n-1} X_c^\top X_c, \qquad X_c = X - \mathbf{1}\mu^\top.
$$

$C$ is symmetric and positive semidefinite, so use a symmetric eigensolver. Sort eigenvalues descending and take the first $k$ eigenvectors as the rows of $W \in \mathbb{R}^{k \times d}$. Scores and the rank-$k$ reconstruction are

$$
Z = X_c W^\top, \qquad \hat X = Z W + \mathbf{1}\mu^\top.
$$

The economy SVD $X_c = U S V^\top$ gives the same axes: the first $k$ rows of $V^\top$, after the same sign fix. The singular values and the eigenvalues of $C$ are related by $\lambda_i = s_i^2 / (n-1)$.

## Resources

- [Essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra) · video · 3Blue1Brown · The eigenvector and change-of-basis episodes are the pictures this lesson assumes.
- [Eigendecomposition](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/eigendecomposition.html) · reading · Dive into Deep Learning · Eigenvalues, symmetric matrices, and why the spectral theorem is the one you want for covariance.
- [Mathematics for Machine Learning](https://mml-book.github.io/book/mml-book.pdf) · book · Deisenroth, Faisal, and Ong · Free PDF. Use the linear algebra chapters and the chapter on dimensionality reduction with PCA.
- [MIT 18.06, Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) · course · Gilbert Strang, MIT OpenCourseWare · Singular values, taught as the structure of any rectangular matrix.
- [Immersive Linear Algebra](http://immersivemath.com/ila/index.html) · reading · Ström, Åström, and Akenine-Möller · Interactive figures for bases and eigenvalues when the algebra gets abstract.

## Exercise

Write `pca_project(X, k)` and `svd_project(X, k)`.

Both take a data matrix `X` shaped `(n, d)` and an integer `k` with `1 <= k <= d`. Both return `(scores, components, mean)`:

- `mean` has shape `(d,)` and is the column mean of `X`.
- `components` has shape `(k, d)`. Each row is a unit vector. Rows are orthonormal. Order them by decreasing variance. Apply the sign convention above.
- `scores` has shape `(n, k)` and equals the centered data times `components.T`.

`pca_project` must build the sample covariance with divisor $n-1$ and call `np.linalg.eigh`. `svd_project` must center the same way and call `np.linalg.svd` on the centered matrix. After the sign fix the two answers agree when the leading singular values are distinct.

Raise `ValueError` if `X` is not 2-D or if `k` is out of range.

```python starter
import numpy as np


def pca_project(X, k):
    raise NotImplementedError


def svd_project(X, k):
    raise NotImplementedError
```

```python solution
import numpy as np


def _sign_fix(components):
    fixed = np.array(components, dtype=float, copy=True)
    for i in range(fixed.shape[0]):
        pivot = int(np.argmax(np.abs(fixed[i])))
        if fixed[i, pivot] < 0:
            fixed[i] *= -1
    return fixed


def _center(X, k):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")
    n, d = X.shape
    if not isinstance(k, (int, np.integer)) or k < 1 or k > d:
        raise ValueError("k must be an integer in 1..d")
    mean = X.mean(axis=0)
    return X - mean, mean, n


def pca_project(X, k):
    centered, mean, n = _center(X, k)
    covariance = (centered.T @ centered) / max(n - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    components = _sign_fix(eigenvectors[:, order[:k]].T)
    scores = centered @ components.T
    return scores, components, mean


def svd_project(X, k):
    centered, mean, _n = _center(X, k)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    components = _sign_fix(vh[:k])
    scores = centered @ components.T
    return scores, components, mean
```

```python tests
import numpy as np

axis = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
scores, components, mean = pca_project(axis, 1)
assert components.shape == (1, 2)
assert np.allclose(components[0], np.array([0.0, 1.0]), atol=1e-8)
assert np.allclose(mean, np.array([0.0, 2.5]))
assert np.allclose(scores.ravel(), np.array([-1.5, -0.5, 0.5, 1.5]), atol=1e-8)

rng = np.random.default_rng(0)
base = rng.normal(size=(60, 3))
cloud = base @ np.diag([3.0, 1.0, 0.4])
s1, c1, m1 = pca_project(cloud, 2)
s2, c2, m2 = svd_project(cloud, 2)
assert np.allclose(c1, c2, atol=1e-6)
assert np.allclose(s1, s2, atol=1e-6)
assert np.allclose(m1, m2, atol=1e-8)
assert np.allclose(c1 @ c1.T, np.eye(2), atol=1e-6)
assert s1.var(axis=0)[0] >= s1.var(axis=0)[1] - 1e-8

full_scores, full_components, full_mean = pca_project(cloud, 3)
recon = full_scores @ full_components + full_mean
assert np.allclose(recon, cloud, atol=1e-8)

try:
    pca_project(np.ones(4), 1)
    raise AssertionError("pca_project should reject a 1-D array")
except ValueError:
    pass
try:
    svd_project(cloud, 0)
    raise AssertionError("k=0 should be rejected")
except ValueError:
    pass
```

## Quiz

### q1
Why does this lesson use `eigh` rather than a general eigensolver on the covariance?

- [ ] Covariance matrices are usually diagonal already.
- [x] Sample covariance is symmetric, so it has real eigenvalues and an orthonormal basis of eigenvectors.
- [ ] `eigh` sorts eigenvalues for you, and a general solver does not.
- [ ] PCA only works on integer matrices.

> Symmetry is the theorem you are using. A general solver is the wrong tool for a matrix you know is symmetric, and the orthonormal basis is exactly the coordinate system PCA wants.

### q2
You keep every principal component of a centered data matrix. What should the reconstruction equal?

- [ ] The column mean, repeated on every row.
- [ ] A matrix of zeros, because the mean was removed.
- [x] The original data matrix.
- [ ] Only the first column of the original matrix.

> The full set of principal axes is a rotation of the centered data. Adding the mean back undoes the centering, so the reconstruction is exact, up to numerical error.

### q3
Two programs return principal axes that are negatives of each other, entry for entry. What is going on?

- [ ] One of them forgot to center the data.
- [ ] The covariance was not symmetric.
- [x] Eigenvectors are only defined up to sign, and the two programs made different choices.
- [ ] Singular values cannot be negative, so one SVD failed.

> $v$ and $-v$ span the same axis and produce the same variance. The sign convention in the exercise is a tie-break for tests, not a new geometric fact.
