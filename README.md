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
- 2000s–2010s: the deep learning revolution — AlexNet, large-scale GPU training, ReLU, dropout, and the ImageNet benchmark.
- Hands-on reproductions: runnable implementations of historically important models, with experiments and comparisons against classical baselines.

Future additions may extend the timeline into modern deep learning, including sequence-to-sequence models, attention, Transformers, diffusion models, and large language models.

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
|   |-- MLP/
|   |   `-- Perceptron-Iris-1958/
|   |       |-- README.md
|   |       |-- train.py
|   |       |-- compare.py
|   |       |-- src/
|   |       `-- results/
|   `-- CNN/
|       |-- LeNet-MNIST-1998/
|       |   |-- README.md
|       |   |-- train.py
|       |   |-- compare.py
|       |   |-- src/
|       |   `-- results/
|       `-- AlexNet-CIFAR10-2012/
|           |-- README.md
|           |-- train.py
|           |-- compare.py
|           |-- setup_data.py
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

### Perceptron & MLP on Iris

The first runnable reproduction is [Perceptron & MLP on Iris](Deep%20Learning/MLP/Perceptron-Iris-1958/README.md), based on:

> Rosenblatt, F. - "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958)

This project implements Rosenblatt's Perceptron and a two-layer MLP in PyTorch, applied to Fisher's Iris dataset (150 samples, 4 features, 3 classes). The purpose is to make the Perceptron convergence theorem concrete: Iris setosa is perfectly linearly separable and the Perceptron classifies it flawlessly, while the versicolor/virginica overlap cannot be resolved without a hidden layer. Adding one hidden layer with ReLU crosses that boundary. The project also compares against SVM, kNN, and Logistic Regression baselines.

### LeNet-5 on MNIST

The first runnable reproduction is [LeNet-5 on MNIST](Deep%20Learning/CNN/LeNet-MNIST-1998/README.md), based on:

> LeCun, Bottou, Bengio, and Haffner - "Gradient-Based Learning Applied to Document Recognition" (1998)

This project implements LeNet-5 in PyTorch and compares it against classical machine learning baselines such as SVM and kNN. The purpose is not only to reproduce a result, but to understand why convolutional networks were such an important shift for vision tasks.

### AlexNet on CIFAR-10

The second reproduction is [AlexNet on CIFAR-10](Deep%20Learning/CNN/AlexNet-CIFAR10-2012/README.md), based on:

> Krizhevsky, Sutskever, and Hinton - "ImageNet Classification with Deep Convolutional Neural Networks" (NeurIPS 2012)

This project adapts AlexNet for the CIFAR-10 dataset (32×32 RGB, 10 classes) and compares it against SVM and kNN baselines. The focus is on understanding what made AlexNet a turning point: ReLU activations, dropout regularization, overlapping pooling, and data augmentation — and why classical methods could not compete at this scale.

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

## Clone Setup (Git LFS Required)

PDF papers and large binary datasets in this repository are stored with [Git LFS](https://git-lfs.com/), not as regular Git blobs. If Git LFS is missing on your machine, clone will leave small pointer files (~130 bytes) instead of real PDFs. Those pointer files start with `version https://git-lfs.github.com/spec/v1` and cannot be opened as PDFs.

### Quick setup (recommended)

Run the setup script for your platform **inside the repository** after clone (or to fix an existing clone):

| Platform | Setup script | Verify only |
|----------|--------------|-------------|
| Windows (PowerShell) | `.\scripts\setup-lfs.ps1` | `.\scripts\verify-lfs.ps1` |
| macOS / Linux | `./scripts/setup-lfs.sh` | `./scripts/verify-lfs.sh` |

On macOS/Linux, make scripts executable once if needed:

```bash
chmod +x scripts/setup-lfs.sh scripts/verify-lfs.sh
```

Each setup script will: install Git LFS if missing, run `git lfs install`, run `git lfs pull`, and verify all PDFs.

---

### Windows

**1. Install Git LFS** (once per machine)

```powershell
winget install GitHub.GitLFS
```

Git LFS is also included with [Git for Windows](https://git-scm.com/download/win).

**2. Clone**

```powershell
git lfs install
git clone https://github.com/dogukannparlak/ai-research-journey.git
cd ai-research-journey
git lfs pull
```

**3. Fix an existing clone**

```powershell
cd ai-research-journey
.\scripts\setup-lfs.ps1
```

**4. Verify**

```powershell
Get-Content "History/2010s/2012/2012 - ImageNet Classification with Deep Convolutional Neural Networks - Krizhevsky et al..pdf" -TotalCount 1
# Expected: %PDF-1.x

.\scripts\verify-lfs.ps1
```

**5. Recommended Git setting**

```powershell
git config --global core.autocrlf false
```

---

### macOS

**1. Install Git LFS** (once per machine)

```bash
brew install git-lfs
```

**2. Clone**

```bash
git lfs install
git clone https://github.com/dogukannparlak/ai-research-journey.git
cd ai-research-journey
git lfs pull
```

**3. Fix an existing clone**

```bash
cd ai-research-journey
chmod +x scripts/setup-lfs.sh scripts/verify-lfs.sh
./scripts/setup-lfs.sh
```

**4. Verify**

```bash
head -n 1 "History/2010s/2012/2012 - ImageNet Classification with Deep Convolutional Neural Networks - Krizhevsky et al..pdf"
# Expected: %PDF-1.x

./scripts/verify-lfs.sh
```

**5. Recommended Git setting**

```bash
git config --global core.autocrlf false
```

---

### Linux

**1. Install Git LFS** (once per machine)

Debian / Ubuntu:

```bash
sudo apt update
sudo apt install git-lfs
```

Fedora:

```bash
sudo dnf install git-lfs
```

Arch:

```bash
sudo pacman -S git-lfs
```

**2. Clone**

```bash
git lfs install
git clone https://github.com/dogukannparlak/ai-research-journey.git
cd ai-research-journey
git lfs pull
```

**3. Fix an existing clone**

```bash
cd ai-research-journey
chmod +x scripts/setup-lfs.sh scripts/verify-lfs.sh
./scripts/setup-lfs.sh
```

**4. Verify**

```bash
head -n 1 "History/2010s/2012/2012 - ImageNet Classification with Deep Convolutional Neural Networks - Krizhevsky et al..pdf"
# Expected: %PDF-1.x

./scripts/verify-lfs.sh
```

**5. Recommended Git setting**

```bash
git config --global core.autocrlf false
```

---

### Adding new PDFs (all platforms)

```bash
git add "History/.../paper.pdf"
git lfs status
git commit -m "Add paper"
git push
```

After `git add`, the staged file should be a small LFS pointer (~130 bytes), not a multi-megabyte blob.

## Philosophy

AI research is easier to understand when it is treated as a sequence of engineering responses to specific failures:

- Perceptrons introduced learnable weights, but struggled with non-linearly separable problems.
- Backpropagation made multilayer learning practical, but deep and recurrent models exposed gradient problems.
- CNNs reduced parameter growth and exploited spatial structure.
- LSTMs introduced controlled memory for long-range dependencies.
- Attention and Transformers later changed how sequence models represent relationships.

This repository follows that chain of problems and solutions, one paper and one implementation at a time.
