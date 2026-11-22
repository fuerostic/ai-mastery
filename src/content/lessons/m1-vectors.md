---
id: vectors
module: math
title: Vectors, norms, and matrices as maps
minutes: 35
summary: A dataset is a list of vectors. A matrix is a linear map that moves them.
idea: The dot product asks how much of one vector lies along another.
---

## Intuition

A row of measurements is a vector whether or not anyone draws an arrow. Age, income, and a count of tokens can sit in the same triple. Once you agree to that, the whole spreadsheet is a list of points in one shared space, and almost every later model is a recipe for moving those points.

The dot product is the question you will ask of two vectors for the rest of this course: how much does one agree with the other? If they point the same way, the product is large and positive. If they are perpendicular, it is zero. Length is the special case of a vector dotted with itself. Cosine similarity throws the lengths away and keeps only the angle, which is why a short sentence and a long paragraph can still "mean" the same direction.

A matrix is not a table you look up. Multiplying by a matrix sends every vector to a new vector, and it does so in a way that respects addition and scaling. Rotation, reflection, stretching, and projection are all linear maps. A neural layer, before the activation, is one of these maps: each output coordinate is a dot product of the input with one row.

Read shapes out loud before you trust a formula. If $A$ is $m \times n$ and $x$ is length $n$, then $Ax$ is length $m$. If that sentence fails, the code will fail too.

## Math

The dot product and the Euclidean norm:

$$
u^\top v = \sum_i u_i v_i, \qquad \|u\|_2 = \sqrt{u^\top u}.
$$

Cosine similarity is the dot product of the unit vectors. By convention in this lesson, a zero vector has similarity 0 with everything, including itself. There is no angle to measure.

$$
\cos\theta = \frac{u^\top v}{\|u\|_2\,\|v\|_2} \quad \text{when both norms are positive.}
$$

A matrix-vector product is a stack of dot products:

$$
(Ax)_i = \sum_j A_{ij} x_j.
$$

Linearity is the property that makes this different from an arbitrary function:

$$
A(x+y) = Ax + Ay, \qquad A(cx) = c\,Ax.
$$

Everything nonlinear in a network — sigmoids, ReLUs, attention softmaxes — is a deliberate break of that rule, inserted between linear maps.

## Resources

- [Essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra) · video · 3Blue1Brown · Geometric pictures of span, basis, and matrices as transformations.
- [Mathematics for Machine Learning](https://mml-book.github.io/) · book · Deisenroth, Faisal, and Ong · The free textbook's linear algebra chapters are the right reference when a proof is missing here.
- [Linear algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html) · reading · Dive into Deep Learning · Short, code-adjacent review of the operations training loops actually use.
- [MIT 18.06, Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) · course · Gilbert Strang, MIT OpenCourseWare · A full first course if the geometry in this lesson feels thin.
- [Immersive Linear Algebra](http://immersivemath.com/ila/index.html) · reading · J. Ström, K. Åström, T. Akenine-Möller · Interactive figures for bases, maps, and eigenvalues when static formulas are not enough.

## Exercise

Implement four functions with NumPy. Shapes are part of the contract.

- `dot(a, b)` returns a Python float.
- `l2_norm(a)` returns a Python float, the Euclidean length.
- `cosine_similarity(a, b)` returns a Python float. If either vector has length 0, return `0.0`.
- `matvec(A, x)` returns a 1-D array. `A` is shaped `(m, n)` and `x` is shaped `(n,)`. Raise `ValueError` when those shapes disagree.

Do not loop over coordinates unless you want to. Elementwise products and `@` are the point.

```python starter
import numpy as np


def dot(a, b):
    raise NotImplementedError


def l2_norm(a):
    raise NotImplementedError


def cosine_similarity(a, b):
    raise NotImplementedError


def matvec(A, x):
    raise NotImplementedError
```

```python solution
import numpy as np


def dot(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sum(a * b))


def l2_norm(a):
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.sum(a * a)))


def cosine_similarity(a, b):
    na = l2_norm(a)
    nb = l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def matvec(A, x):
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    if A.ndim != 2 or x.ndim != 1 or A.shape[1] != x.shape[0]:
        raise ValueError("A must have shape (m, n) and x shape (n,)")
    return A @ x
```

```python tests
import numpy as np

assert abs(dot([1, 2, 3], [4, 5, 6]) - 32) < 1e-9
assert abs(l2_norm([3, 4]) - 5) < 1e-9
assert abs(l2_norm([0, 0, 0])) < 1e-12
assert abs(cosine_similarity([1.0, 0.0], [1.0, 1.0]) - 1 / np.sqrt(2)) < 1e-9
assert cosine_similarity([0.0, 0.0], [4.0, -1.0]) == 0.0
assert abs(cosine_similarity([1.0, 0.0], [-2.0, 0.0]) - (-1.0)) < 1e-9

A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
x = np.array([1.0, -1.0])
assert np.allclose(matvec(A, x), np.array([-1.0, -1.0, -1.0]))
try:
    matvec(np.ones((2, 3)), np.ones(2))
    raise AssertionError("matvec should reject a shape mismatch")
except ValueError:
    pass
```

## Quiz

### q1
A matrix $A$ with shape $(4, 3)$ multiplies a vector $x$. Which shape must $x$ have, and what is the shape of $Ax$?

- [ ] $x$ is $(4,)$ and $Ax$ is $(3,)$.
- [x] $x$ is $(3,)$ and $Ax$ is $(4,)$.
- [ ] $x$ is $(4, 3)$ and $Ax$ is a scalar.
- [ ] $x$ can have either shape; broadcasting will fix it.

> Each row of $A$ is dotted with $x$, so $x$ must have one entry per column. Four rows produce four outputs.

### q2
Two document vectors point in exactly the same direction, but one is ten times longer. What is their cosine similarity?

- [ ] 10
- [ ] 0
- [x] 1
- [ ] It depends on the dimension.

> Cosine similarity divides by both lengths, so shared direction gives 1 and opposite direction gives $-1$. The extra length is exactly what the formula removes.

### q3
Which pair of rules is the definition of a linear map?

- [ ] $Ax$ is always at least as long as $x$, and $A$ is square.
- [ ] $A(xy) = Ax\,Ay$, and $A(x+y) = Ax + Ay$.
- [x] $A(x+y) = Ax + Ay$, and $A(cx) = c\,Ax$.
- [ ] $A$ has orthonormal columns, and $A^\top A = I$.

> Linearity is additivity and scaling. Products of vectors, shape, and length are separate ideas. A projection is linear and it shortens vectors.