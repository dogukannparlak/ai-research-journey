# ai-research-journey

A personal research archive for studying the history of artificial intelligence and machine learning through the papers, architectures, and experiments that shaped the field.

The goal of this repository is not to collect every important paper. It is to follow the evolution of ideas carefully: why certain architectures appeared, what problems they solved, where they failed, and how later work built on top of them.

The guiding principle is simple:

> Read the paper. Understand the math. Rebuild the idea. Compare it with context.

## What This Repository Covers

This journey starts with early neural computation and connectionism, then moves toward the foundations of deep learning:

- Pre-1980 foundations: McCulloch-Pitts neurons, Rosenblatt's Perceptron, early learning systems, and the first debates around connectionist models.
- 1980s revival: Hopfield networks, backpropagation, universal approximation, and recurrent learning.
- 1990s foundations: vanishing gradients, convolutional networks, LSTM, bidirectional RNNs, wake-sleep learning, and LeNet.
- Hands-on reproductions: runnable implementations of historically important models, with experiments and comparisons against classical baselines.

Future additions may extend the timeline into modern deep learning, including AlexNet, sequence-to-sequence models, attention, Transformers, diffusion models, and large language models.

## Repository Structure

```text
ai-research-journey/
|-- History/
|   |-- Pre-1980/
|   |   `-- Pre-1980-Notes.md
|   |-- 1980s/
|   |   `-- 1980s-Notes.md
|   `-- 1990s/
|       `-- 1990s-Notes.md
|
|-- Deep Learning/
|   `-- CNN/
|       `-- LeNet-MNIST-1998/
|           |-- README.md
|           |-- train.py
|           |-- compare.py
|           |-- src/
|           `-- results/
|
`-- README.md
```

## Current Projects

### Historical Notes

The `History/` directory contains period-based notes that connect the technical ideas to their historical motivation:

- [Pre-1980 Notes](History/Pre-1980/Pre-1980-Notes.md)
- [1980s Notes](History/1980s/1980s-Notes.md)
- [1990s Notes](History/1990s/1990s-Notes.md)

These notes focus on architecture-level understanding: what each model assumes, how information flows through it, how learning happens, and what limitations pushed the field forward.

### LeNet-5 on MNIST

The first runnable reproduction is [LeNet-5 on MNIST](Deep%20Learning/CNN/LeNet-MNIST-1998/README.md), based on:

> LeCun, Bottou, Bengio, and Haffner - "Gradient-Based Learning Applied to Document Recognition" (1998)

This project implements LeNet-5 in PyTorch and compares it against classical machine learning baselines such as SVM and kNN. The purpose is not only to reproduce a result, but to understand why convolutional networks were such an important shift for vision tasks.

## Research Method

Each topic is studied through four layers:

1. Historical context - what problem existed at the time?
2. Mathematical structure - what is the core formulation?
3. Architectural idea - how does the model process information?
4. Practical verification - what happens when the idea is implemented and tested?

This makes the repository both a reading archive and an experimental lab.

## How To Use This Repository

Start with the historical notes if you want the conceptual timeline. Start with the implementation folders if you want executable experiments.

For runnable projects, each subdirectory includes its own README with setup instructions, architecture notes, results, and reproduction steps.

## Philosophy

AI research is easier to understand when it is treated as a sequence of engineering responses to specific failures:

- Perceptrons introduced learnable weights, but struggled with non-linearly separable problems.
- Backpropagation made multilayer learning practical, but deep and recurrent models exposed gradient problems.
- CNNs reduced parameter growth and exploited spatial structure.
- LSTMs introduced controlled memory for long-range dependencies.
- Attention and Transformers later changed how sequence models represent relationships.

This repository follows that chain of problems and solutions, one paper and one implementation at a time.
