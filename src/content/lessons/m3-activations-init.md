---
id: activations-init
module: nets
title: Initialization and vanishing gradients
minutes: 35
summary: A deep stack multiplies local derivatives. The product is the whole story.
idea: Vanishing gradients are a product of numbers smaller than one, repeated.
---

## Intuition

A deep network's gradient with respect to an early weight is a product. Each layer contributes a weight and a local activation derivative. If the typical factor is 0.2, then twenty layers contribute $0.2^{20}$, which is not a small gradient so much as a zero gradient. Learning in the early layers stops because the loss, from their point of view, does not change.

Sigmoid is the usual villain in this story, and it deserves the reputation. Its derivative is $\sigma(z)(1-\sigma(z))$, which peaks at $1/4$ when $z = 0$ and is tiny once $|z|$ is large. A product of several $1/4$ factors is already a vanishing gradient, even before the weights join in. ReLU's local factor is 0 or 1. It does not systematically shrink, though it can die: a unit that is negative for every example in the batch sends no gradient and receives no update.

Initialization is how you keep the pre-activations in the region where the local factor is healthy. Glorot normal, also called Xavier normal, draws weights with standard deviation $\sqrt{2 / (\mathrm{fan\_in} + \mathrm{fan\_out})}$. It is the usual scale for tanh and sigmoid, whose derivatives are largest near zero. He normal, $\sqrt{2 / \mathrm{fan\_in}}$, is the usual scale for ReLU, as in the previous lesson. Neither formula trains the network. They stop the first forward pass from saturating it.

The exercise is the product itself, plus the two formulas. Once you can compute the backward scale of a sigmoid chain and a ReLU chain, the plots in the recommended notes are no longer mysterious.

## Math

Sigmoid and its derivative:

$$
\sigma(z) = \frac{1}{1+e^{-z}}, \qquad \sigma'(z) = \sigma(z)\big(1-\sigma(z)\big) \le \frac{1}{4}.
$$

A scalar chain $h_{\ell} = \phi(w_\ell h_{\ell-1})$ sends a gradient backward multiplied by $w_\ell \phi'(z_\ell)$ at each layer. The end-to-end factor is

$$
\prod_\ell w_\ell \, \phi'(z_\ell).
$$

Glorot normal and He normal standard deviations:

$$
\sigma_{\mathrm{Glorot}} = \sqrt{\frac{2}{\mathrm{fan\_in}+\mathrm{fan\_out}}}, \qquad
\sigma_{\mathrm{He}} = \sqrt{\frac{2}{\mathrm{fan\_in}}}.
$$

These are standard deviations for a normal draw, not the bounds of a uniform draw. The uniform Glorot bound $\sqrt{6/(\mathrm{fan\_in}+\mathrm{fan\_out})}$ is a different formula.

## Resources

- [CS231n: neural networks, part 2](https://cs231n.github.io/neural-networks-2/) · reading · Stanford CS231n · Initialization, activation statistics, and batch normalization, with the plots this lesson is about.
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) · video · Andrej Karpathy · The lectures keep returning to dead neurons and bad initial scales.
- [Neural networks](https://www.3blue1brown.com/topics/neural-networks) · video · 3Blue1Brown · A visual pass over what a layer is doing to gradients.
- [Deep Learning, chapter on multilayer networks](https://www.deeplearningbook.org/contents/mlp.html) · book · Goodfellow, Bengio, and Courville · Hidden units, architecture, and backpropagation in the standard reference.
- [A recipe for training neural networks](http://karpathy.github.io/2019/04/25/recipe/) · reading · Andrej Karpathy · Practical checks for a network whose gradients have quietly gone to zero.

## Exercise

Implement the two initialization scales and the backward product.

- `xavier_std(fan_in, fan_out)` returns $\sqrt{2/(\mathrm{fan\_in}+\mathrm{fan\_out})}$ as a Python float. This is the Glorot normal standard deviation.
- `he_std(fan_in)` returns $\sqrt{2/\mathrm{fan\_in}}$ as a Python float.
- `backprop_product(kind, weights, preacts)` multiplies $w \cdot \phi'(z)$ across the paired sequences. `kind` is `"sigmoid"` or `"relu"`. ReLU's derivative is 1 when $z > 0$ and 0 otherwise. Raise `ValueError` for any other kind. Return a Python float. The sequences may be lists or arrays of the same length.

```python starter
import numpy as np


def xavier_std(fan_in, fan_out):
    raise NotImplementedError


def he_std(fan_in):
    raise NotImplementedError


def backprop_product(kind, weights, preacts):
    raise NotImplementedError
```

```python solution
import numpy as np


def xavier_std(fan_in, fan_out):
    return float(np.sqrt(2.0 / (fan_in + fan_out)))


def he_std(fan_in):
    return float(np.sqrt(2.0 / fan_in))


def backprop_product(kind, weights, preacts):
    if kind not in ("sigmoid", "relu"):
        raise ValueError("kind must be 'sigmoid' or 'relu'")
    product = 1.0
    for weight, pre in zip(weights, preacts):
        z = float(pre)
        if kind == "sigmoid":
            activated = 1.0 / (1.0 + np.exp(-z))
            local = activated * (1.0 - activated)
        else:
            local = 1.0 if z > 0.0 else 0.0
        product *= float(weight) * local
    return float(product)
```

```python tests
import numpy as np

assert abs(xavier_std(2, 2) - np.sqrt(0.5)) < 1e-12
assert abs(he_std(4) - np.sqrt(0.5)) < 1e-12
assert abs(he_std(8) - np.sqrt(0.25)) < 1e-12

# sigma'(0) = 0.25, so two layers of weight 1 at pre-activation 0 give 1/16.
assert abs(backprop_product("sigmoid", [1.0, 1.0], [0.0, 0.0]) - 0.0625) < 1e-12
assert backprop_product("relu", [0.5, 2.0], [1.0, -0.2]) == 0.0
assert abs(backprop_product("relu", [0.5, -3.0], [1.0, 4.0]) - (0.5 * -3.0)) < 1e-12
try:
    backprop_product("tanh", [1.0], [0.0])
    raise AssertionError("unknown activations should be rejected")
except ValueError:
    pass
```

## Quiz

### q1
The maximum of $\sigma'(z)$ is

- [ ] 1, at $z = 0$.
- [x] $1/4$, at $z = 0$.
- [ ] $1/2$, at $z = 1$.
- [ ] unbounded as $z$ grows.

> $\sigma(1-\sigma)$ is a parabola in $\sigma$ on $[0,1]$, and it peaks at $\sigma = 1/2$, where the value is $1/4$. That peak is the best a sigmoid layer will ever do.

### q2
Glorot normal and He normal differ mainly because

- [ ] He is for biases and Glorot is for weights.
- [x] ReLU deletes the negative half of a symmetric pre-activation, so the weight scale has to be larger to keep the variance steady.
- [ ] Glorot is a uniform bound and He is always $\sqrt{6}$.
- [ ] He initialization sets half of the weights to zero on purpose.

> The extra factor of 2 in He's fan-in formula replaces the variance that ReLU throws away. Sigmoid and tanh do not throw half of the mass away in that same way, and they saturate, which is a separate problem.

### q3
A ReLU unit whose pre-activation is negative for every example in the batch

- [ ] still receives a small gradient from $\sigma'(z)$.
- [x] receives a zero parameter gradient from those examples and will not move unless some other example wakes it up.
- [ ] is numerically identical to a sigmoid unit.
- [ ] increases the condition number of the Hessian by 2.

> The local factor is 0, so the chain stops. This is a dead unit, and it is why a huge learning rate on the first step can permanently remove capacity.
