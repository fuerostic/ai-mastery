import type { ModuleMeta } from "./types";

/** Curriculum order. Lesson prose and exercises live in src/content/lessons. */
export const modulesMeta: ModuleMeta[] = [
  {
    id: "math",
    number: 1,
    title: "Math foundations for machine learning",
    tagline: "Geometry, calculus, probability, and the shape of a training problem.",
    description:
      "Training code is linear algebra plus a derivative plus a probability model. This module makes those four languages usable: vectors as data, spectra as structure, gradients as local linear maps, likelihood as a loss, and step size as a statement about curvature.",
    lessonIds: ["vectors", "eigen-pca-svd", "gradients", "probability-mle", "optimization"],
  },
  {
    id: "classic",
    number: 2,
    title: "NumPy and classical learning",
    tagline: "Array notation, then two models you can derive on paper.",
    description:
      "NumPy is the notation. Logistic regression and softmax are the first models whose loss, gradient, and training loop you should be able to write without a framework. Edit distance is here because dynamic programming keeps returning in sequence models and tokenizers.",
    lessonIds: ["numpy-thinking", "logistic-regression", "softmax-cross-entropy", "edit-distance"],
  },
  {
    id: "nets",
    number: 3,
    title: "Neural nets from scratch",
    tagline: "A derivative engine small enough to read, then a network trained with it.",
    description:
      "Build the chain rule into a tiny value type, backpropagate a multilayer perceptron by hand, fit a problem a linear model cannot represent, and see how initialization decides whether those gradients survive depth.",
    lessonIds: ["micrograd", "mlp-backprop", "train-mlp", "activations-init"],
  },
  {
    id: "transformers",
    number: 4,
    title: "Transformers",
    tagline: "From characters to a causal attention block.",
    description:
      "Tokenization, embeddings, positions, scaled dot-product attention, and multi-head attention. The goal is to be able to read a GPT block and know which tensor each line is talking about.",
    lessonIds: ["bpe", "embeddings-positions", "scaled-attention", "multihead-gpt"],
  },
  {
    id: "training",
    number: 5,
    title: "Training language models",
    tagline: "The pretraining loss, how it scales, and how a finished model gets adapted.",
    description:
      "Next-token prediction is the whole pretraining objective. Scaling laws say how that loss moves with compute. Supervised fine-tuning, preference optimization, LoRA, mixtures of experts, and distillation are the tools people use after a base model exists.",
    lessonIds: ["next-token", "scaling-laws", "sft-dpo", "lora-moe-distill"],
  },
  {
    id: "research",
    number: 6,
    title: "Research skills for NLP",
    tagline: "Make a number mean something.",
    description:
      "Macro-F1, resampling, annotator disagreement, dumb baselines, and contamination checks. A leaderboard score is an estimate. This module is about saying what the estimate is, and when it is not evidence.",
    lessonIds: ["macro-f1", "significance", "annotation-kappa", "baselines-contamination"],
  },
];
