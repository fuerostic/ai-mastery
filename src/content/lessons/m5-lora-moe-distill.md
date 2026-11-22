---
id: lora-moe-distill
module: training
title: LoRA, mixtures of experts, and distillation
minutes: 45
summary: Three ways to change a big model without retraining every weight the same way.
idea: Adaptation, sparsity, and a student can all be written as small linear algebra on top of a frozen model.
---

## Intuition

Full fine-tuning writes a new value into every parameter. That is simple and expensive, and it produces a separate copy of a model that was already large. Low-rank adaptation freezes $W$ and learns a low-rank update $\Delta W = AB$. If $W$ is $d_{\mathrm{in}} \times d_{\mathrm{out}}$ and the rank $r$ is small, $A$ is $d_{\mathrm{in}} \times r$ and $B$ is $r \times d_{\mathrm{out}}$. The forward pass adds $\mathrm{scale} \cdot xAB$ to the original $xW$. People often set the scale to $\alpha / r$. The point of the exercise is the forward formula, not a particular $\alpha$.

A mixture of experts changes the width of the MLP, not the training copy. Several feed-forward networks sit side by side. A router looks at $x$ and produces a distribution over experts. A dense mixture is the weighted sum of every expert. A sparse mixture, which is what Switch and the large MoE models use, keeps only the top expert or the top few and sends the token down that path. The exercise implements the dense weighted sum, because that is the differentiable object the sparse version approximates. The router in a real MoE is itself trained, and load-balancing losses exist so every expert gets some work. Those losses are in the papers, not in the function you are writing.

Distillation trains a student to match a teacher. Hinton's version softens both distributions with a temperature $T$, takes the KL divergence from teacher to student, and multiplies by $T^2$ so the gradient scale stays comparable as $T$ changes. At $T = 1$ you are matching ordinary probabilities. At higher $T$ the teacher's runner-up classes become visible, which is the information a hard label throws away. If the student already matches the teacher, the loss is zero at any temperature.

## Math

LoRA, with row-vector inputs and a frozen $W$:

$$
y = xW + \mathrm{scale}\, xAB, \qquad A \in \mathbb{R}^{d_{\mathrm{in}} \times r}, \; B \in \mathbb{R}^{r \times d_{\mathrm{out}}}.
$$

Dense MoE, with router logits $r(x)$ and expert outputs $f_e(x)$:

$$
g = \mathrm{softmax}(r(x)), \qquad y = \sum_e g_e f_e(x).
$$

Distillation, averaged over the batch, with $p_T = \mathrm{softmax}(z / T)$:

$$
L = T^2 \, \mathrm{KL}\big(p_T^{\mathrm{teacher}} \,\|\, p_T^{\mathrm{student}}\big)
= T^2 \sum_j p_{T,j}^{\mathrm{teacher}} \log \frac{p_{T,j}^{\mathrm{teacher}}}{p_{T,j}^{\mathrm{student}}}.
$$

## Resources

- [LoRA: low-rank adaptation of large language models](https://arxiv.org/abs/2106.09685) · paper · Hu et al. · The low-rank update and the argument that adaptation lives in a small subspace.
- [LoRA conceptual guide](https://huggingface.co/docs/peft/conceptual_guides/lora) · reading · Hugging Face PEFT · How the forward pass is implemented, including the $\alpha / r$ scale.
- [QLoRA](https://arxiv.org/abs/2305.14314) · paper · Dettmers et al. · The same low-rank update on a quantized frozen model.
- [Parameter-efficient finetuning with LoRA](https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-E) · code · Sebastian Raschka · A from-scratch LoRA appendix on the GPT built in the book.
- [Outrageously large neural networks: the sparsely-gated mixture-of-experts layer](https://arxiv.org/abs/1701.06538) · paper · Shazeer et al. · The MoE layer and the load-balancing problem.
- [Switch Transformers](https://arxiv.org/abs/2101.03961) · paper · Fedus, Zoph, and Shazeer · Top-1 routing at a scale where the dense mixture no longer fits.
- [Distilling the knowledge in a neural network](https://arxiv.org/abs/1503.02531) · paper · Hinton, Vinyals, and Dean · Temperature, soft targets, and the $T^2$ factor.

## Exercise

Implement the three forward formulas.

- `lora_forward(x, W, A, B, scale)` uses the shapes above. `x` is `(n, d_in)`. Return `y` shaped `(n, d_out)`. Raise `ValueError` if `A.shape[1] != B.shape[0]`.
- `mixture_of_experts(router_logits, expert_outputs)` takes router logits `(n, e)` and expert outputs `(n, e, d)`. Return `(y, gates)` where `gates` is the row-wise softmax and `y` is the gated sum over experts, shaped `(n, d)`.
- `distillation_loss(student_logits, teacher_logits, temperature=1.0)` takes two `(n, V)` arrays. Return the mean $T^2 \mathrm{KL}(\mathrm{teacher}\|\mathrm{student})$ as a Python float, with both distributions softened by `temperature`.

```python starter
import numpy as np


def lora_forward(x, W, A, B, scale):
    raise NotImplementedError


def mixture_of_experts(router_logits, expert_outputs):
    raise NotImplementedError


def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    raise NotImplementedError
```

```python solution
import numpy as np


def lora_forward(x, W, A, B, scale):
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape[1] != B.shape[0]:
        raise ValueError("A.shape[1] must equal B.shape[0]")
    return x @ W + float(scale) * (x @ A @ B)


def mixture_of_experts(router_logits, expert_outputs):
    logits = np.asarray(router_logits, dtype=float)
    experts = np.asarray(expert_outputs, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    gates = np.exp(shifted)
    gates = gates / np.sum(gates, axis=1, keepdims=True)
    mixed = np.sum(gates[:, :, None] * experts, axis=1)
    return mixed, gates


def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    temperature = float(temperature)

    def log_softmax(logits):
        z = np.asarray(logits, dtype=float) / temperature
        z = z - np.max(z, axis=-1, keepdims=True)
        return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))

    log_student = log_softmax(student_logits)
    log_teacher = log_softmax(teacher_logits)
    teacher = np.exp(log_teacher)
    kl = np.sum(teacher * (log_teacher - log_student), axis=-1)
    return float(np.mean(kl) * (temperature ** 2))
```

```python tests
import math
import numpy as np

x = np.array([[1.0, 0.0]])
W = np.zeros((2, 2))
A = np.array([[1.0], [0.0]])
B = np.array([[0.0, 2.0]])
assert np.allclose(lora_forward(x, W, A, B, scale=0.5), np.array([[0.0, 1.0]]))
try:
    lora_forward(x, W, np.ones((2, 2)), np.ones((3, 2)), 1.0)
    raise AssertionError("rank mismatch should be rejected")
except ValueError:
    pass

logits = np.array([[0.0, 0.0]])
experts = np.array([[[1.0, 0.0], [3.0, 4.0]]])
mixed, gates = mixture_of_experts(logits, experts)
assert np.allclose(gates, np.array([[0.5, 0.5]]))
assert np.allclose(mixed, np.array([[2.0, 2.0]]))

def oracle(student, teacher, temperature):
    def probs(row):
        scaled = [value / temperature for value in row]
        peak = max(scaled)
        weights = [math.exp(value - peak) for value in scaled]
        total = sum(weights)
        return [weight / total for weight in weights]
    total = 0.0
    for s_row, t_row in zip(student, teacher):
        ps, pt = probs(s_row), probs(t_row)
        total += sum(p * math.log(p / q) for p, q in zip(pt, ps))
    return (total / len(student)) * (temperature ** 2)

student = [[0.0, 4.0], [1.0, 1.0]]
teacher = [[0.0, 0.0], [1.0, 1.0]]
for temperature in (1.0, 2.0):
    got = distillation_loss(np.array(student), np.array(teacher), temperature=temperature)
    assert abs(got - oracle(student, teacher, temperature)) < 1e-8
assert abs(distillation_loss(np.array([[0.2, -0.4, 0.1]]), np.array([[0.2, -0.4, 0.1]]), temperature=3.0)) < 1e-8
```

## Quiz

### q1
A LoRA update of rank $r$ on a $d \times d$ weight uses about

- [ ] $d^2$ new parameters, the same as full fine-tuning.
- [x] $2dr$ new parameters, plus the frozen original matrix.
- [ ] $r$ new parameters, one per singular value.
- [ ] $d$ new parameters, a vector of scales.

> $A$ contributes $d r$ and $B$ contributes $r d$. The original $W$ is not trained.

### q2
The dense mixture in this lesson differs from a Switch-style MoE because

- [ ] Switch uses a softmax and the dense mixture does not.
- [x] every expert contributes to every token here. Switch routes each token to one expert.
- [ ] dense mixtures have no router.
- [ ] distillation requires the dense form.

> The softmax gate is shared by both stories. Sparsity is the decision to multiply the gate by a mask and skip the other experts.

### q3
The $T^2$ factor in the distillation loss

- [ ] converts KL divergence into cross-entropy.
- [x] keeps the gradient from shrinking just because a higher temperature flattened both distributions.
- [ ] is only correct at $T = 1$, where it equals 1.
- [ ] replaces the teacher's softmax.

> Dividing logits by $T$ flattens the distribution and shrinks the raw KL. Multiplying back by $T^2$ is the correction Hinton et al. use so different temperatures stay on a comparable scale.
