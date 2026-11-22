---
id: optimization
module: math
title: Gradient descent, condition numbers, and Adam
minutes: 40
summary: A step size is a bet on curvature. Adam makes that bet separately in each coordinate.
idea: A skinny valley, not a mysterious loss, is why one learning rate fails.
---

## Intuition

Gradient descent is the gradient lesson with a decision attached. At the current point, trust the local linear approximation enough to step against the slope, then look again:

$$
x \leftarrow x - \eta \nabla f(x).
$$

The step size $\eta$ is how far you trust that line. Too far, and you leave the region where the line was honest. Too near, and you spend the afternoon crossing a gentle slope.

On a quadratic bowl the honesty of the line is controlled by the Hessian. If every direction curves the same amount, one $\eta$ is a good compromise. If one eigenvalue is huge and another is tiny, the level sets are long thin ellipses. A step that is stable on the steep wall is tiny along the floor of the valley. The condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ measures that mismatch. Preconditioning, normalization, and adaptive optimizers are all attempts to make $\kappa$ less embarrassing.

Convexity is the promise that the local picture is also the global one. A function is convex when every chord sits above the graph. For a smooth function that is the same as a Hessian that is positive semidefinite everywhere. Gradient descent on a smooth convex function, with a small enough constant step, approaches a global minimizer. Neural losses are not convex. They have saddle points and flat regions, and the step you use is a local heuristic that has worked often enough to become the default. Knowing the convex case tells you what the heuristic is borrowing.

Adam keeps two running averages. The first is momentum: an exponential average of the gradient, so coordinates that keep pointing the same way accumulate a larger step. The second is an average of squared gradients, used as a per-coordinate scale. Both averages start at zero, which would shrink the early steps, so Adam divides by a bias-correction term that is near zero at step 1 and near one later. From zero moments, the first Adam step has length about $\eta$ in every coordinate. The adaptation appears after that.

## Math

For $f(x) = \tfrac12 x^\top H x$ with $H$ symmetric positive definite, the error multiplies by $I - \eta H$ each step of gradient descent. The iteration stays stable when $0 < \eta < 2/\lambda_{\max}$. The rate then depends on $\kappa = \lambda_{\max}/\lambda_{\min}$.

A function is convex when, for every $x, y$ and every $t \in [0,1]$,

$$
f(tx + (1-t)y) \le t f(x) + (1-t) f(y).
$$

Adam's moments and the bias-corrected step, with $g_t = \nabla f(x_t)$, are

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2}, \\
\hat m_t &= \frac{m_t}{1-\beta_1^t}, \qquad
\hat v_t = \frac{v_t}{1-\beta_2^t}, \\
x_{t+1} &= x_t - \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon).
\end{aligned}
$$

This lesson's code returns the vector that is subtracted, $\eta\, \hat m \oslash (\sqrt{\hat v}+\epsilon)$, together with the new moments. It does not update $x$ itself.

## Resources

- [Gradient descent](https://d2l.ai/chapter_optimization/gd.html) · reading · Dive into Deep Learning · The update, the convex quadratic, and what a step size is doing.
- [Adam](https://d2l.ai/chapter_optimization/adam.html) · reading · Dive into Deep Learning · Momentum and RMSProp combined, with the bias correction written out.
- [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) · book · Boyd and Vandenberghe · The free book. Convexity, condition numbers, and why the nice theorems stop where they stop.
- [EE364a: Convex Optimization I](https://web.stanford.edu/class/ee364a/) · course · Stanford · Stephen Boyd's course page, with the same book.
- [CS231n: Optimization](https://cs231n.github.io/optimization-1/) · reading · Stanford CS231n · Learning rates, momentum, and the visual behavior of the common optimizers.
- [CS231n: Optimization, part 2](https://cs231n.github.io/optimization-2/) · reading · Stanford CS231n · Second-order intuition and the practical tricks that sit on top of gradient descent.

## Exercise

Implement plain gradient descent and one Adam step.

- `gradient_descent(grad_fn, x0, lr, steps)` starts at `x0`. `grad_fn` takes a 1-D array and returns a gradient of the same shape. Return `(final_x, trajectory)` where `trajectory[0]` is the start and the length is `steps + 1`. Do not mutate `x0`.
- `adam_update(grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)` takes the current gradient and the previous moments. `t` starts at 1. Return `(new_m, new_v, step)`, where `step` is the vector you would subtract from the parameters.

```python starter
import numpy as np


def gradient_descent(grad_fn, x0, lr, steps):
    raise NotImplementedError


def adam_update(grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    raise NotImplementedError
```

```python solution
import numpy as np


def gradient_descent(grad_fn, x0, lr, steps):
    x = np.asarray(x0, dtype=float).copy()
    trajectory = [x.copy()]
    for _ in range(int(steps)):
        x = x - float(lr) * np.asarray(grad_fn(x), dtype=float)
        trajectory.append(x.copy())
    return x, trajectory


def adam_update(grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    grad = np.asarray(grad, dtype=float)
    m = beta1 * np.asarray(m, dtype=float) + (1.0 - beta1) * grad
    v = beta2 * np.asarray(v, dtype=float) + (1.0 - beta2) * (grad ** 2)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    step = float(lr) * m_hat / (np.sqrt(v_hat) + eps)
    return m, v, step
```

```python tests
import numpy as np

final, trajectory = gradient_descent(lambda x: 2.0 * x, np.array([10.0]), lr=0.1, steps=20)
expected = 10.0 * (0.8 ** 20)
assert abs(final[0] - expected) < 1e-8
assert len(trajectory) == 21
assert abs(trajectory[0][0] - 10.0) < 1e-12
assert abs(trajectory[1][0] - 8.0) < 1e-12
start = np.array([10.0])
gradient_descent(lambda x: 2.0 * x, start, lr=0.1, steps=1)
assert abs(start[0] - 10.0) < 1e-12

m, v, step = adam_update(np.array([1.0, -2.0]), np.zeros(2), np.zeros(2), t=1, lr=0.001)
assert np.allclose(m, np.array([0.1, -0.2]), atol=1e-12)
assert np.allclose(v, np.array([0.001, 0.004]), atol=1e-12)
assert np.allclose(step, np.array([0.001, -0.001]), atol=1e-8)

m2, v2, step2 = adam_update(np.array([1.0]), m[:1], v[:1], t=2, lr=0.001)
assert np.allclose(step2, np.array([0.001]), atol=1e-8)
```

## Quiz

### q1
Gradient descent on $f(x) = x^2$ uses the update $x \leftarrow x - \eta \cdot 2x$. Which step size keeps the iteration from blowing up?

- [ ] Any $\eta > 0$. The function is convex, so divergence is impossible.
- [x] $0 < \eta < 1$.
- [ ] $\eta > 1$, otherwise the step is smaller than the gradient.
- [ ] Only $\eta = 1$, because that reaches zero in one step.

> The error multiplies by $1 - 2\eta$. That factor has absolute value below 1 when $\eta$ is between 0 and 1. $\eta = 1$ does reach zero in one step, but it is not the only stable choice. Convexity does not license an arbitrary step.

### q2
A quadratic has Hessian eigenvalues $100$ and $0.1$. What is the condition number, and why does it matter?

- [ ] $10$, the difference of the eigenvalues. It sets the optimal batch size.
- [x] $1000$, the ratio of the eigenvalues. A single step size cannot suit both directions well.
- [ ] $0.001$, the reciprocal ratio. Adam ignores it.
- [ ] It is undefined, because the Hessian is not square.

> $\kappa = \lambda_{\max}/\lambda_{\min} = 1000$. The steep direction demands a small step. The flat direction then crawls.

### q3
Adam is started with $m_0 = 0$, $v_0 = 0$, and a gradient whose coordinates are all nonzero. The first subtracted step is, approximately,

- [ ] the zero vector, because the moments were initialized at zero.
- [ ] $\eta$ times the raw gradient, with no rescaling.
- [x] a vector of length about $\eta$ in every coordinate, with the sign of the gradient.
- [ ] the Hessian-vector product $H g$.

> Bias correction divides the first moment by $1-\beta_1$ and the second by $1-\beta_2$, which undoes the $(1-\beta)$ factors from a zero start. You are left with $\hat m = g$ and $\hat v = g^{\odot 2}$, so each coordinate is scaled to about $\pm \eta$.
