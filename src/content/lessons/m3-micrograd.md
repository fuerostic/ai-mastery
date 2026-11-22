---
id: micrograd
module: nets
title: A scalar autograd engine
minutes: 45
summary: The chain rule, stored on the values that participate in it.
idea: Backpropagation is bookkeeping. Each node remembers how to blame its inputs.
---

## Intuition

A training step needs one number, the loss, and one derivative of that number for every parameter. You can write those derivatives by hand for a two-layer network. You will not enjoy writing them for a transformer. Automatic differentiation keeps the hand algebra local. Each operation knows its own derivative. The chain rule multiplies those local factors in the right order.

Build a value that stores a scalar and a gradient. Addition and multiplication create new values and remember their parents. The local rules are short. The derivative of $a+b$ with respect to either parent is 1. The derivative of $a \cdot b$ with respect to $a$ is $b$. ReLU passes the gradient through when the input was positive and stops it when the input was negative. At zero, this lesson sends zero backward.

`backward` walks the graph in an order where every child is finished before its parents are asked to absorb blame. A topological sort does that. Set the loss gradient to 1, then visit nodes in reverse. If a value is used twice, its gradient is the sum of the two paths. That is the multivariable chain rule, and it falls out of the additions if you do not overwrite a gradient when you mean to accumulate it.

Call `backward` twice and you should get the same gradients, not twice the gradients. Zero every gradient in the graph before you accumulate. The starter already forwards left-hand floats through `__radd__` and `__rmul__`. Implement `__add__`, `__mul__`, `relu`, and `backward`.

## Math

For a parent $p$ of an output $o$,

$$
\frac{\partial L}{\partial p} \mathrel{+}= \frac{\partial L}{\partial o} \frac{\partial o}{\partial p}.
$$

Local factors:

$$
\frac{\partial (a+b)}{\partial a} = 1, \qquad
\frac{\partial (ab)}{\partial a} = b, \qquad
\frac{\partial \mathrm{ReLU}(x)}{\partial x} = \begin{cases} 1 & x > 0 \\ 0 & x \le 0. \end{cases}
$$

If $z = x \cdot x$, both parents are the same object. The two paths each contribute $x$, and the total is $2x$, which is the ordinary derivative of $x^2$. The backward closure must add into `grad` rather than assign it. A set of unique parents does not remove the second path: the closure was created twice, once for each use.

## Resources

- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) · video · Andrej Karpathy · The micrograd lecture is the long version of this lesson. Watch it with the code open.
- [micrograd](https://github.com/karpathy/micrograd) · code · Andrej Karpathy · A tiny scalar autograd engine. Build yours first, then compare the graph walk.
- [Neural Networks and Deep Learning, chapter 2](http://neuralnetworksanddeeplearning.com/chap2.html) · book · Michael Nielsen · Backpropagation with the index notation written out slowly.
- [Forward and backward propagation](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html) · reading · Dive into Deep Learning · The chain rule drawn as a computational graph.
- [A recipe for training neural networks](http://karpathy.github.io/2019/04/25/recipe/) · reading · Andrej Karpathy · What to do once the gradients exist and a network still will not train.

## Exercise

Complete the `Value` class. `__init__`, `__radd__`, and `__rmul__` are given. Store children in `_prev` when you create an output, and store a local `_backward` closure that adds into the parents' gradients.

`relu` returns a new `Value` whose data is `max(data, 0)`. `backward` builds a topological order, zeros every `.grad` in that order, sets the output gradient to 1, then calls `_backward` in reverse order.

```python starter
class Value:
    def __init__(self, data, _children=()):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def relu(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
```

```python solution
class Value:
    def __init__(self, data, _children=()):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def relu(self):
        out = Value(self.data if self.data > 0.0 else 0.0, (self,))

        def _backward():
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        seen = set()

        def build(node):
            if node not in seen:
                seen.add(node)
                for parent in node._prev:
                    build(parent)
                topo.append(node)

        build(self)
        for node in topo:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

```python tests
a = Value(2.0)
b = Value(-3.0)
c = Value(4.0)
loss = a * b + c
loss.backward()
assert abs(loss.data - (-2.0)) < 1e-9
assert abs(a.grad - (-3.0)) < 1e-9
assert abs(b.grad - 2.0) < 1e-9
assert abs(c.grad - 1.0) < 1e-9
loss.backward()
assert abs(a.grad - (-3.0)) < 1e-9

x = Value(2.0)
y = (x * x).relu()
y.backward()
assert abs(y.data - 4.0) < 1e-9
assert abs(x.grad - 4.0) < 1e-9

negative = Value(-2.0)
stopped = negative.relu()
stopped.backward()
assert stopped.data == 0.0
assert negative.grad == 0.0

x = Value(1.5)
mixed = 3 + x * 2
mixed.backward()
assert abs(mixed.data - 6.0) < 1e-9
assert abs(x.grad - 2.0) < 1e-9

x = Value(3.0)
y = Value(-1.0)
z = (x + y).relu() * x
z.backward()
assert abs(z.data - 6.0) < 1e-9
assert abs(x.grad - 5.0) < 1e-9
assert abs(y.grad - 3.0) < 1e-9
```

## Quiz

### q1
Why does `backward` zero gradients before accumulating?

- [ ] A topological sort is undefined if any gradient is nonzero.
- [x] A second call would otherwise add another copy of the same derivative on top of the first.
- [ ] ReLU is undefined at a nonzero gradient.
- [ ] The chain rule only applies when every parent gradient starts at 1.

> The closures use `+=`. Leftover gradients from a previous call are not part of the new derivative.

### q2
`z = x * x`, with `x.data = 3`, and `z.backward()`. What is `x.grad`?

- [ ] 3
- [ ] 9
- [x] 6
- [ ] 0, because both parents are the same object and the set keeps only one.

> The derivative of $x^2$ is $2x$. The implementation gets there by adding $x$ twice, once for each use in the product.

### q3
ReLU's local gradient at a negative input is 0 because

- [ ] the chain rule does not apply to piecewise functions.
- [x] the function is locally constant there, so a small wiggle in the input does not change the output.
- [ ] negative activations are numerically unsafe to differentiate.
- [ ] the topological order skips negative nodes.

> To the right of zero, ReLU is the identity and the local factor is 1. To the left it is the zero function and the local factor is 0. This lesson breaks the tie at zero by sending no gradient.
