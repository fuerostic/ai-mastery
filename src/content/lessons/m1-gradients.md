---
id: gradients
module: math
title: Gradients, Jacobians, and Hessians
minutes: 40
summary: The derivative is a local linear map. The gradient, Jacobian, and Hessian are its coordinates.
idea: Training trusts a tiny linear approximation and then steps along it.
---

## Intuition

A derivative is the slope of the best tiny straight line through a function. In several variables that line becomes a tiny linear map, and the gradient is the vector of its slopes. It points toward the steepest ascent of a scalar loss. Gradient descent walks the other way.

If the function returns a vector — a whole layer of activations — the local linear map is a matrix. That matrix is the Jacobian. Each row is the gradient of one output. Backpropagation is the chain rule applied to a stack of these maps, multiplying Jacobians without ever writing the full matrix down.

The Hessian is one derivative further: the Jacobian of the gradient. Its entries are curvatures. A positive definite Hessian is a local bowl. A Hessian with both positive and negative eigenvalues is a saddle. You will almost never build this matrix for a neural net, because it is enormous. You still feel it. A long thin valley is a Hessian whose eigenvalues differ by orders of magnitude, and that shape is why a single step size can be too timid in one direction and explosive in another.

Check an analytic gradient against a central finite difference before you trust a training loop. If they disagree, the bug is almost always in the algebra, not in the optimizer.

## Math

For $f: \mathbb{R}^d \to \mathbb{R}$,

$$
(\nabla f(x))_i = \frac{\partial f}{\partial x_i}.
$$

The quadratic that shows up in least squares and in local approximations is $f(x) = x^\top A x + b^\top x$. Differentiating carefully, without assuming $A$ is symmetric, gives

$$
\nabla f(x) = (A + A^\top) x + b.
$$

If $A$ happens to be symmetric this is $2Ax + b$. The factor of two is the usual place to drop a term and then fail a gradient check.

The Jacobian of $g: \mathbb{R}^n \to \mathbb{R}^m$ has entries $J_{ij} = \partial g_i / \partial x_j$. The Hessian of a scalar $f$ has entries $H_{ij} = \partial^2 f / \partial x_i \partial x_j$. Mixed partials agree when $f$ is smooth enough, so $H$ is symmetric and `eigh` applies.

A central difference along coordinate $i$, with a small $\epsilon$, is

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}.
$$

The two-sided formula cancels the even part of the error. One-sided differences are noisier for the same $\epsilon$.

## Resources

- [Essence of calculus](https://www.3blue1brown.com/topics/calculus) · video · 3Blue1Brown · Derivatives as local linear approximations, which is the only picture you need before the multivariable case.
- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/index.html) · reading · Parr and Howard · A careful layout of gradients, Jacobians, and the layout conventions that cause bugs.
- [Calculus](https://d2l.ai/chapter_preliminaries/calculus.html) · reading · Dive into Deep Learning · Partial derivatives and the chain rule, written next to the code that uses them.
- [Mathematics for Machine Learning](https://mml-book.github.io/book/mml-book.pdf) · book · Deisenroth, Faisal, and Ong · The vector-calculus chapter is the right place to slow down.
- [Single-variable calculus review](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/single-variable-calculus.html) · reading · Dive into Deep Learning · A compact reminder of the one-dimensional facts the multivariable formulas extend.

## Exercise

Implement the quadratic, its analytic gradient, and a central-difference gradient.

- `quadratic(x, A, b)` returns a Python float, $x^\top A x + b^\top x$. Do not assume $A$ is symmetric.
- `quadratic_grad(x, A, b)` returns the gradient $(A + A^\top)x + b$ as a 1-D array.
- `numerical_grad(f, x, eps=1e-6)` returns a central-difference estimate of $\nabla f(x)$. `f` takes a 1-D array and returns a scalar. Do not mutate the caller's vector.

```python starter
import numpy as np


def quadratic(x, A, b):
    raise NotImplementedError


def quadratic_grad(x, A, b):
    raise NotImplementedError


def numerical_grad(f, x, eps=1e-6):
    raise NotImplementedError
```

```python solution
import numpy as np


def quadratic(x, A, b):
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(x @ A @ x + b @ x)


def quadratic_grad(x, A, b):
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    return (A + A.T) @ x + b


def numerical_grad(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        plus = x.copy()
        minus = x.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (float(f(plus)) - float(f(minus))) / (2.0 * eps)
    return grad
```

```python tests
import numpy as np

A = np.array([[2.0, 0.5], [0.0, 3.0]])
b = np.array([-1.0, 2.0])
x = np.array([0.3, -0.7])
assert abs(quadratic(x, A, b) - (-0.155)) < 1e-9
assert np.allclose(quadratic_grad(x, A, b), np.array([-0.15, -2.05]), atol=1e-9)

estimate = numerical_grad(lambda z: quadratic(z, A, b), x)
assert np.allclose(estimate, quadratic_grad(x, A, b), atol=1e-6)

symmetric = np.array([[2.0, 0.3], [0.3, 1.0]])
point = np.array([1.5, -0.4])
analytic = quadratic_grad(point, symmetric, np.zeros(2))
assert np.allclose(analytic, 2 * symmetric @ point, atol=1e-9)
numeric = numerical_grad(lambda z: quadratic(z, symmetric, np.array([0.2, -0.5])), point, eps=1e-5)
assert np.allclose(numeric, quadratic_grad(point, symmetric, np.array([0.2, -0.5])), atol=1e-6)
```

## Quiz

### q1
$f(x) = x^\top A x$ and $A$ is symmetric. Which expression is $\nabla f(x)$?

- [ ] $Ax$
- [x] $2Ax$
- [ ] $A^\top A x$
- [ ] $x^\top A$

> Each factor of $x$ contributes an $Ax$ when $A$ is symmetric. Forgetting the 2 is the classic gradient-check failure on a quadratic bowl.

### q2
What does a central finite difference estimate?

- [ ] The Hessian's largest eigenvalue, exactly.
- [x] A single partial derivative, by probing the function a little above and below the point.
- [ ] The global minimum of any smooth function.
- [ ] The Jacobian of a matrix inverse.

> You only learn the slope at one coordinate. Repeating it for every coordinate builds a numerical gradient. It is a test of an analytic formula, not a replacement for one inside a large training loop.

### q3
A loss has a Hessian with eigenvalues $1000$ and $0.01$ at the current point. What does that say about the local bowl?

- [ ] The loss is non-differentiable there.
- [ ] Every direction has the same curvature.
- [x] The bowl is a long thin valley: very curved in one direction, almost flat in another.
- [ ] Gradient descent will diverge for every step size.

> The ratio of those eigenvalues is the condition number. It is the reason a single learning rate struggles, which is the subject of the optimization lesson.
