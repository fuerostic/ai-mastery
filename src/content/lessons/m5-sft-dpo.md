---
id: sft-dpo
module: training
title: Supervised fine-tuning and direct preference optimization
minutes: 40
summary: Same next-token loss on demonstrations, then a classification loss on pairs of answers.
idea: DPO is a preference loss with a closed form, not a second copy of reinforcement learning you have to implement first.
---

## Intuition

Pretraining makes a model that continues text. It does not make a model that answers a question in the format you wanted. Supervised fine-tuning is the same next-token loss on a narrower dataset: prompts paired with demonstrations of the behavior. You are not teaching a new objective. You are changing which continuations are common. The practical details that matter are which tokens you mask. Typically the prompt tokens are ignored and the assistant tokens are scored, so the model is not trained to reproduce the user's wording.

Preference data is a different shape. For a prompt $x$ you have two answers, and a person preferred one. Reinforcement learning from human feedback fits a reward model to those pairs and then runs a policy-gradient algorithm with a penalty that keeps the policy near the reference model. That works, and it is a lot of machinery. Direct preference optimization starts from the same Bradley-Terry preference model plus the same KL penalty, writes down the optimal policy in closed form, and turns the result into a binary classification loss on the two answers. There is no explicit reward head and no rollout loop in the loss you will implement.

The loss looks at four log-probabilities: the policy and the reference model, each on the chosen and rejected answers. If the policy likes the chosen answer more than the rejected one, relative to how the reference model felt, the loss is small. If the policy copies the reference, the margin is zero and the loss is $\log 2$, the same number as a coin flip. $\beta$ is the temperature of that tradeoff. Larger $\beta$ punishes departure from the reference more heavily.

## Math

The DPO loss for one pair, up to the expectation, is

$$
L = -\log \sigma\!\left(\beta \left[
\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
-
\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
\right]\right).
$$

In log-probabilities that is

$$
z = \beta \big( (\log \pi_\theta(y_w) - \log \pi_\theta(y_l)) - (\log \pi_{\mathrm{ref}}(y_w) - \log \pi_{\mathrm{ref}}(y_l)) \big),
$$

$$
L = \log(1 + e^{-z}) = \mathrm{softplus}(-z).
$$

Average $L$ over the batch. Use `logaddexp(0, -z)` so a large positive margin underflows to a zero loss instead of overflowing the exponential. This exercise receives log-probabilities that have already been summed over tokens. It does not recompute a sequence log-prob.

## Resources

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) · paper · Ouyang et al. · InstructGPT: the SFT, reward-model, and RLHF pipeline this lesson is simplifying.
- [Direct preference optimization](https://arxiv.org/abs/2305.18290) · paper · Rafailov et al. · The derivation of the classification loss from the KL-constrained reward objective.
- [Illustrating reinforcement learning from human feedback](https://huggingface.co/blog/rlhf) · reading · Hugging Face · The three-stage RLHF picture, before DPO removes two of the stages.
- [Supervised fine-tuning](https://huggingface.co/learn/llm-course/chapter11/1) · course · Hugging Face LLM Course · What an SFT dataset looks like and which tokens are scored.
- [Finetuned language models are zero-shot learners](https://arxiv.org/abs/2109.01652) · paper · Wei et al. · Instruction tuning at scale, the FLAN result.
- [LLMs from scratch, chapter 7](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07) · code · Sebastian Raschka · Finetuning to follow instructions, on a model you built in earlier chapters.
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) · book · Sutton and Barto · The free RL book, if the policy-gradient side of RLHF is the part you do not yet have.

## Exercise

Implement the batched DPO loss.

`dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta=0.1)` takes four 1-D arrays of equal length. Each entry is a sequence log-probability. Return the mean softplus loss as a Python float. Raise `ValueError` if the shapes differ.

When the policy matches the reference on both answers, the loss equals $\log 2$. A large positive margin drives the loss toward 0.

```python starter
import numpy as np


def dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta=0.1):
    raise NotImplementedError
```

```python solution
import numpy as np


def dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta=0.1):
    logp_w = np.asarray(logp_w, dtype=float)
    logp_l = np.asarray(logp_l, dtype=float)
    logp_ref_w = np.asarray(logp_ref_w, dtype=float)
    logp_ref_l = np.asarray(logp_ref_l, dtype=float)
    if not (logp_w.shape == logp_l.shape == logp_ref_w.shape == logp_ref_l.shape):
        raise ValueError("all four log-probability arrays must share a shape")
    if logp_w.ndim != 1:
        raise ValueError("log-probabilities must be 1-D")
    margin = (logp_w - logp_l) - (logp_ref_w - logp_ref_l)
    return float(np.mean(np.logaddexp(0.0, -beta * margin)))
```

```python tests
import numpy as np

coin = dpo_loss([0.0, -1.0], [0.0, -1.0], [0.0, -1.0], [0.0, -1.0], beta=0.1)
assert abs(coin - np.log(2)) < 1e-9
moved = dpo_loss([10.0], [0.0], [0.0], [0.0], beta=0.1)
assert abs(moved - np.logaddexp(0.0, -1.0)) < 1e-9
assert moved < coin
try:
    dpo_loss([0.0, 1.0], [0.0], [0.0], [0.0])
    raise AssertionError("shape mismatches should be rejected")
except ValueError:
    pass
```

## Quiz

### q1
Supervised fine-tuning, as used before RLHF or DPO, optimizes

- [ ] a new reinforcement-learning objective with a learned reward.
- [x] next-token cross-entropy on demonstrations, often with the prompt tokens masked out of the loss.
- [ ] a contrastive loss between chosen and rejected answers.
- [ ] the KL divergence to the pretraining corpus.

> The objective does not change. The dataset does, and the mask decides which tokens are demonstrations rather than context.

### q2
If the policy assigns the same log-probabilities as the reference model, the DPO loss is

- [ ] 0, because the KL penalty is satisfied.
- [x] $\log 2$, the loss of a fair coin deciding which answer was preferred.
- [ ] $\beta$, independent of the data.
- [ ] undefined, because the margin denominator is zero.

> The margin inside the sigmoid is zero. $-\log \sigma(0) = \log 2$. Matching the reference is not the same as ranking the preferred answer higher.

### q3
$\beta$ in the DPO loss

- [ ] is the learning rate of the reward model.
- [x] scales how hard the policy must prefer the chosen answer in order to overcome the pull toward the reference.
- [ ] must be 1, or the Bradley-Terry model is invalid.
- [ ] replaces the need for a reference model when it is set to 0. The formula still contains the reference.

> The reference terms are inside the margin either way. $\beta$ sets the stiffness of the preference. Setting it to 0 makes the loss a constant, which is not a useful special case.
