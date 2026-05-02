# AlexNet on CIFAR-10 — Krizhevsky et al. (2012)

> A PyTorch replication of **"ImageNet Classification with Deep Convolutional Neural Networks"**  
> *Krizhevsky, Sutskever & Hinton — NeurIPS 2012*

This project adapts AlexNet for CIFAR-10 (32×32 RGB images, 10 classes) and benchmarks it against classical machine learning methods (SVM, kNN) — contextualising the structural argument of the paper: that learned hierarchical features fundamentally outperform hand-crafted representations at the scale of real-world visual recognition.

---

## Table of Contents

- [Historical Context](#historical-context)
- [Project Structure](#project-structure)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Codebase Walkthrough](#codebase-walkthrough)
- [Results](#results)
- [Reproducing the Experiment](#reproducing-the-experiment)
- [Design Decisions & Deviations from the Paper](#design-decisions--deviations-from-the-paper)

---

## Historical Context

| Year | Milestone |
|------|-----------|
| 1998 | LeNet-5 — *Gradient-Based Learning Applied to Document Recognition* (MNIST, grayscale digits) |
| 2009 | Krizhevsky — *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10 introduced) |
| 2010 | First ImageNet Large Scale Visual Recognition Challenge (ILSVRC); best top-5 error: ~28% |
| 2011 | ILSVRC best: ~26% — all traditional computer vision (SIFT, Fisher Vectors, SVMs) |
| **2012** | **AlexNet wins ILSVRC with 15.3% top-5 error — runner-up at 26.2%** |
| 2014 | VGGNet — systematic depth study; 19-layer networks |
| 2015 | ResNet — residual connections enable 100+ layer training |

By 2012, computer vision research had plateaued. The dominant pipeline — hand-craft features (SIFT, HOG) → encode (Fisher Vector, VLAD) → classify (SVM) — had been refined for a decade with diminishing returns. AlexNet did not just win the ImageNet competition: it won by a margin so large (10+ percentage points) that the entire field was forced to reconsider its assumptions within months.

The paper's core argument was straightforward: convolutional networks trained on enough data and compute can learn representations that are strictly better than anything a human can engineer. CIFAR-10 — a smaller, more tractable benchmark — allows the same architectural principles to be studied and reproduced without requiring multi-GPU ImageNet training.

---

## Project Structure

```
AlexNet-CIFAR10-2012/
│
├── config.py           — Central configuration: paths, hyperparameters, class names
├── train.py            — Entry point: train AlexNet end-to-end
├── compare.py          — Benchmark: AlexNet vs SVM vs kNN
├── setup_data.py       — Extract cifar-10-python.tar.gz before first run
├── requirements.txt
│
├── src/
│   ├── model.py        — AlexNet architecture (adapted for 32×32 inputs)
│   ├── dataset.py      — CIFAR-10 pickle loader, normalisation, augmentation
│   ├── trainer.py      — Training loop with LR scheduler
│   ├── evaluator.py    — Test accuracy + per-class breakdown + confusion matrix
│   ├── visualizer.py   — Training curves, confusion matrices, sample predictions
│   └── classical.py    — LinearSVC and kNN baselines (sklearn)
│
├── data/
│   ├── raw/
│   │   └── cifar-10-python.tar.gz
│   └── extracted/
│       └── cifar-10-batches-py/    (created by setup_data.py)
├── models/
│   └── alexnet.pt      — Saved model weights (after training)
└── results/
    ├── alexnet_training_history.png
    ├── alexnet_confusion_matrix.png
    ├── alexnet_sample_predictions.png
    ├── svm_confusion_matrix.png
    ├── knn_confusion_matrix.png
    └── comparison.csv
```

---

## Architecture Deep Dive

AlexNet is structured in two conceptually separate stages: a **feature extractor** (five convolutional blocks) that learns spatial hierarchies from raw pixels, and a **classifier** (three fully connected layers) that maps extracted features to class probabilities.

The original architecture targets 224×224 ImageNet images. This implementation adapts the spatial dimensions for CIFAR-10's 32×32 inputs while preserving the architectural logic of each block.

```
Input
  (3 × 32 × 32)
        │
  ┌─────▼─────┐
  │  Block 1  │  Conv2d(3 → 64, kernel=3×3, pad=1) + ReLU    →  64 × 32 × 32
  │           │  MaxPool2d(3×3, stride=2, pad=1)              →  64 × 16 × 16
  └─────┬─────┘
  ┌─────▼─────┐
  │  Block 2  │  Conv2d(64 → 192, kernel=5×5, pad=2) + ReLU  →  192 × 16 × 16
  │           │  MaxPool2d(3×3, stride=2, pad=1)              →  192 × 8 × 8
  └─────┬─────┘
  ┌─────▼─────┐
  │  Block 3  │  Conv2d(192 → 384, kernel=3×3, pad=1) + ReLU →  384 × 8 × 8
  └─────┬─────┘
  ┌─────▼─────┐
  │  Block 4  │  Conv2d(384 → 256, kernel=3×3, pad=1) + ReLU →  256 × 8 × 8
  └─────┬─────┘
  ┌─────▼─────┐
  │  Block 5  │  Conv2d(256 → 256, kernel=3×3, pad=1) + ReLU →  256 × 8 × 8
  │           │  MaxPool2d(3×3, stride=2, pad=1)              →  256 × 4 × 4
  └─────┬─────┘
        │  Flatten  →  4096
  ┌─────▼─────┐
  │  FC  1    │  Dropout(0.5) → Linear(4096 → 1024) + ReLU
  └─────┬─────┘
  ┌─────▼─────┐
  │  FC  2    │  Dropout(0.5) → Linear(1024 → 512) + ReLU
  └─────┬─────┘
  ┌─────▼─────┐
  │  Output   │  Linear(512 → 10)
  └───────────┘
```

**Total trainable parameters: ~7,173,450**

Compare this to LeNet-5's 61,706 parameters — AlexNet is over 100× larger, and that scale difference is precisely the point of the paper.

### Why These Choices?

| Component | Paper Choice | This Implementation | Rationale |
|-----------|-------------|---------------------|-----------|
| Activation | `ReLU` | `ReLU` | Preserved exactly — central contribution of the paper |
| Pooling | `MaxPool` (overlapping, 3×3, stride 2) | Same | Preserved; overlapping pooling reduces error ~0.3–0.4% |
| Conv1 kernel | 11×11, stride 4 | 3×3, stride 1 | 11×11 stride 4 on 32×32 collapses to 6×6 immediately; 3×3 preserves spatial information |
| LRN | Local Response Normalisation | Omitted | Shown to be ineffective in later work; Batch Norm replaced it entirely |
| FC width | 4096 → 4096 | 1024 → 512 | Proportionally reduced for 32×32 inputs; 4096 units on a 4×4 feature map would be severely overparameterised |
| Dropout | 0.5 in FC layers | 0.5 in FC layers | Preserved exactly — introduced in this paper as a regularisation technique |
| Training hardware | 2× GTX 580 GPUs | Single CPU/GPU | Single-machine reproduction; accuracy is not hardware-dependent |

---

## Codebase Walkthrough

### `config.py` — Central Configuration

All hyperparameters, filesystem paths, and dataset constants live in one place. The CIFAR-10 class name list is stored here so that every module (`evaluator.py`, `visualizer.py`, `compare.py`) references the same source of truth rather than duplicating strings.

```python
BATCH_SIZE    = 128
LEARNING_RATE = 0.001
EPOCHS        = 30
DEVICE        = "cpu"
NUM_CLASSES   = 10
IMAGE_SIZE    = 32

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
```

Batch size is set to 128 (double LeNet's 64) because CIFAR-10 is more complex — larger batches provide more stable gradient estimates, which matters more when learning from colour images with 10 diverse object categories.

---

### `src/dataset.py` — Data Pipeline

**Raw format of CIFAR-10:**  
The dataset ships as five Python pickle files (`data_batch_1` through `data_batch_5`) plus `test_batch`. Each file contains a dict with:

- `b"data"`: a `(10000, 3072)` NumPy array — 10,000 images, each stored as 3072 flat values (1024 red + 1024 green + 1024 blue)
- `b"labels"`: a list of 10,000 integers in `[0, 9]`

`_load_batch()` reads and reshapes this into `(N, 3, 32, 32)` — the channel-first format PyTorch expects.

**`CIFAR10Dataset`** applies normalisation using channel-wise statistics computed over the entire training set:

```python
_MEAN = (0.4914, 0.4822, 0.4465)   # per-channel mean
_STD  = (0.2470, 0.2435, 0.2616)   # per-channel std
```

These values shift and scale each channel independently so that inputs to the network have approximately zero mean and unit variance — stabilising gradient flow in early training.

**Data augmentation** is applied exclusively during training:

- `RandomHorizontalFlip` — a cat facing left is still a cat; symmetric classes benefit from this
- `RandomCrop(32, padding=4)` — pads the image by 4 pixels on each side, then randomly crops back to 32×32; the model sees slightly different spatial positions each epoch

Augmentation is deliberately *not* applied during validation or testing. This is critical: augmentation is a training-time regularisation strategy, not a data transformation. Applying it during evaluation would make metrics non-reproducible.

**`load_data()`** handles the full split:

```
50,000 training samples
  └─ 45,000 used for training
  └─  5,000 held out for validation (10%)
10,000 test samples (never seen during training)
```

**`load_raw_arrays()`** returns the unnormalised `(N, 3, 32, 32)` arrays for use by `classical.py`, which handles its own preprocessing independently of the PyTorch pipeline.

---

### `src/model.py` — AlexNet Implementation

The model is split into two `nn.Sequential` blocks — `features` and `classifier` — mirroring the paper's conceptual distinction between the convolutional representation-learning stages and the dense decision stages.

```python
def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)   # flatten: (B, 256, 4, 4) → (B, 4096)
    return self.classifier(x)
```

**On ReLU:**  
LeNet-5 used `Tanh`. AlexNet switched to `ReLU (inplace=True)`. The difference is significant:

- `Tanh` saturates at ±1, producing near-zero gradients for large activations (vanishing gradients)
- `ReLU` has a constant gradient of 1 for positive inputs — gradients flow back through deep networks without decay
- `inplace=True` modifies the tensor in-place, reducing memory allocation by ~50% with no accuracy cost

**On Dropout:**  
The paper introduced Dropout as a regularisation technique. Without it, the large FC layers (millions of parameters) would memorise the training set. By randomly zeroing half the activations during each forward pass, the network cannot rely on any single neuron — it is forced to learn redundant representations. Dropout is only active during `model.train()` and is automatically disabled during `model.eval()`.

`count_parameters()` sums all `requires_grad=True` parameters — useful for verifying that the adapted architecture has a reasonable parameter count for the input size.

---

### `src/trainer.py` — Training Loop

The `Trainer` class encapsulates the full training procedure. The internal `_run_epoch()` method is shared between training and validation passes — the `training` flag controls:

1. `model.train()` vs `model.eval()` — enables/disables Dropout
2. `torch.set_grad_enabled(training)` — disables autograd graph construction during validation, saving memory and compute

**Loss, Optimiser, and Scheduler:**

- `CrossEntropyLoss` — standard for multi-class classification; combines `LogSoftmax` + `NLLLoss`
- `Adam` (lr=0.001) — deviation from the paper's SGD with momentum; converges reliably without hand-tuned learning rate schedules
- `ReduceLROnPlateau` — if validation loss does not improve for 5 consecutive epochs, the learning rate is multiplied by 0.1. This allows training to proceed at a coarser scale initially and fine-tune as the loss landscape flattens

The scheduler is stepped **after** the validation pass each epoch:
```python
self.scheduler.step(val_loss)
```

Per-epoch metrics (loss and accuracy for both train and val) are stored in a `history` dict and returned to `train.py` for plotting.

---

### `src/evaluator.py` — Test Evaluation

Runs inference over the entire test set in batches, accumulates predictions, then computes:

1. **Overall accuracy** — fraction of correctly classified samples across all 10,000 test images
2. **Per-class accuracy** — breakdown by category, revealing which classes the model struggles with
3. **Confusion matrix** — a 10×10 integer matrix where `matrix[true][predicted]` counts predictions; diagonal entries are correct; off-diagonal entries reveal systematic confusions

The per-class breakdown is the key diagnostic for CIFAR-10: high accuracy on `airplane` but low accuracy on `cat` points to specific representation failures, not general model weakness.

Typical hard confusions on CIFAR-10: **cat↔dog**, **automobile↔truck**, **deer↔horse** — these pairs share similar colour distributions and silhouette shapes at 32×32 resolution.

---

### `src/visualizer.py` — Result Plots

Three plots are generated and saved to `results/` as PNG files.

**1. Training History**  
Side-by-side line charts of loss and accuracy for train and validation sets across epochs. A growing gap between the two curves signals overfitting; a flat val_acc curve despite decreasing train_loss indicates the learning rate should be reduced.

**2. Confusion Matrix**  
A heatmap using CIFAR-10 class name labels on both axes. Because CIFAR-10 images are colour photographs (not grayscale digits), class-level confusions are more visually interpretable — seeing `cat` confused with `dog` is immediately meaningful in a way that `4` confused with `9` is not.

**3. Sample Predictions**  
A 2×5 grid of colour test images. Unlike the MNIST visualiser which displays grayscale tensors directly, CIFAR-10 images require **denormalisation** before display:

```python
img = images[i].permute(1, 2, 0).numpy()
img = (img * _STD + _MEAN).clip(0, 1)
```

`permute(1, 2, 0)` converts channel-first `(3, 32, 32)` to channel-last `(32, 32, 3)` for `matplotlib`. The inverse normalisation restores the original pixel value range. Without this step, displayed images would have incorrect colours and visible clipping artefacts.

---

### `src/classical.py` — Classical ML Baselines

The input dimensionality for CIFAR-10 is `3 × 32 × 32 = 3072` — compared to MNIST's `784`. Classical methods receive flat, normalised RGB pixel vectors. `StandardScaler` is applied before both SVM and kNN for the same reasons as in the LeNet project: distance-based methods require zero-mean unit-variance inputs to behave sensibly.

**SVM (`run_svm`):**  
Uses `LinearSVC` with `C=0.1`. The challenge on CIFAR-10 is substantially harder than MNIST: handwritten digits have low within-class variance (a '3' always looks like a '3'), while natural images have enormous within-class variance (cats appear in countless poses, scales, and lighting conditions). Linear separability over raw pixels is a much stronger assumption on natural images.

**kNN (`run_knn`):**  
k is set to 5 (vs 3 for MNIST) because CIFAR-10's larger intra-class variance means single nearest neighbours are less reliable. Inference is slow — scanning 50,000 training vectors per test sample in 3072-dimensional space. `n_jobs=-1` parallelises across CPU cores.

---

### `setup_data.py` — Data Extraction

Run once before training. Extracts `data/raw/cifar-10-python.tar.gz` into `data/extracted/cifar-10-batches-py/`. The script checks whether the target directory already contains files to avoid redundant extraction.

---

### `train.py` — Orchestration

The entry point wires all components together in the correct sequence:

```
setup: load_data()  →  AlexNet()  →  Trainer.train()  →  Trainer.save()
                                           ↓
                                    Evaluator.evaluate()
                                           ↓
                         plot_training_history()
                         plot_confusion_matrix()
                         plot_sample_predictions()
```

---

### `compare.py` — Benchmark Script

Designed to be run **after** `train.py`. It:

1. Loads saved `alexnet.pt` weights — no retraining needed.
2. Trains SVM and kNN from scratch on raw arrays.
3. Prints a formatted comparison table and writes `results/comparison.csv`.

---

## Results

### Model Comparison — CIFAR-10 Test Set

> Results from a completed training run: 30 epochs, Adam (lr=0.001), RTX 3060 Laptop GPU,
> batch size 128, with random crop and horizontal flip augmentation.

| Model | Test Accuracy | Error Rate | Train Time |
|-------|:------------:|:----------:|:----------:|
| **AlexNet** | **79.31%** | **20.69%** | ~25 min (RTX 3060, 30 epochs) |
| SVM (LinearSVC) | TBD | TBD | run `compare.py` |
| kNN (k=5) | TBD | TBD | run `compare.py` |

### Per-Class Breakdown

| Class | Correct | Total | Accuracy | Note |
|-------|:-------:|:-----:|:--------:|------|
| ship | 922 | 1000 | **92.2%** | Best — distinct silhouette and blue background |
| automobile | 889 | 1000 | **88.9%** | Strong — consistent shape and colour patterns |
| truck | 858 | 1000 | **85.8%** | Strong — similar to automobile |
| frog | 856 | 1000 | **85.6%** | Strong — distinctive green colouring |
| deer | 793 | 1000 | 79.3% | |
| airplane | 788 | 1000 | 78.8% | |
| horse | 787 | 1000 | 78.7% | |
| bird | 701 | 1000 | 70.1% | Confused with airplane and deer |
| dog | 698 | 1000 | 69.8% | Confused with cat |
| cat | 639 | 1000 | **63.9%** | Worst — visually similar to dog at 32×32 |

### Training Curve Summary

| Phase | Epoch 1 | Epoch 10 | Epoch 20 | Epoch 30 |
|-------|:-------:|:--------:|:--------:|:--------:|
| Train Loss | 1.843 | 0.882 | 0.685 | 0.598 |
| Train Acc | 28.9% | 69.4% | 76.7% | 79.7% |
| Val Loss | 1.559 | 0.877 | 0.729 | 0.655 |
| Val Acc | 40.0% | 69.6% | 74.8% | 78.1% |

The learning rate remained constant at `1e-03` throughout all 30 epochs — `ReduceLROnPlateau` did not trigger because validation loss was still consistently improving, never plateauing for 5 consecutive epochs. This indicates that extended training would likely yield further accuracy gains.

### Paper vs Replication

| Method | Top-5 Error (paper, ImageNet) | CIFAR-10 Accuracy (ours) |
|--------|:-----------------------------:|:------------------------:|
| kNN (raw pixels) | — | TBD (run `compare.py`) |
| SVM (Linear) | — | TBD (run `compare.py`) |
| **AlexNet** | **15.3%** | **79.31%** |

Direct comparison between ImageNet top-5 error and CIFAR-10 accuracy is not meaningful — the datasets differ in scale, resolution, and number of classes. The relevant comparison is the **gap between AlexNet and the classical baselines**. Run `compare.py` to populate the table above.

### Key Observations

- **79.31% overall accuracy** on 10,000 unseen test images across 10 natural image categories, trained from scratch with no pretraining.
- The learning rate scheduler never fired across all 30 epochs, which is a positive sign: the model was still learning consistently. Extended training (50–100 epochs) would likely push accuracy above 82%.
- **cat (63.9%)** is the hardest class by a clear margin. At 32×32 resolution, cats and dogs share nearly identical low-level statistics — fur texture, face shape, colouring. This confusion is structural, not a model failure. It reflects a fundamental limitation of the input resolution.
- **ship (92.2%)** is the easiest class. Ocean backgrounds provide strong discriminative signal that the convolutional layers exploit effectively even in early training epochs.
- The train/val accuracy gap at epoch 30 is 1.6 points (79.7% vs 78.1%), indicating that overfitting is mild and the augmentation strategy is working as intended.
- **bird (70.1%)** underperforms relative to other animal classes. Birds appear in highly variable poses — perched, in flight, facing different directions — making consistent feature extraction harder at 32×32 resolution.
- The gap between training accuracy (79.7%) and final test accuracy (79.31%) is under 0.5 points, confirming that the model generalises well to unseen data.

---

## Reproducing the Experiment

### Prerequisites

```bash
pip install -r requirements.txt
```

```
numpy
matplotlib
torch
torchvision
scikit-learn
```

### Data

Place the CIFAR-10 archive at `data/raw/cifar-10-python.tar.gz`, then extract:

```bash
python setup_data.py
```

This creates `data/extracted/cifar-10-batches-py/` containing:
- `data_batch_1` through `data_batch_5` — 50,000 training images
- `test_batch` — 10,000 test images
- `batches.meta` — class name mappings

### Training

```bash
# Train AlexNet (~2 hours on CPU for 30 epochs)
python train.py
```

Output: `models/alexnet.pt`, training history and evaluation plots in `results/`.

### Comparison

```bash
# Run full benchmark (requires trained alexnet.pt)
python compare.py
```

Output: `results/comparison.csv`, per-model confusion matrix plots.

---

## Design Decisions & Deviations from the Paper

| Aspect | Original Paper (2012) | This Implementation | Reason |
|--------|----------------------|---------------------|--------|
| Input size | 224×224 (ImageNet) | 32×32 (CIFAR-10) | Tractable reproduction on a single machine |
| Conv1 kernel | 11×11, stride 4 | 3×3, stride 1 | 11×11 on 32×32 collapses spatial dims to 6×6 in one step |
| FC layer width | 4096 → 4096 | 1024 → 512 | Proportionally scaled for the smaller feature map (4×4 vs 6×6) |
| LRN | Local Response Normalisation | Omitted | Superseded by BatchNorm; no longer considered beneficial |
| Optimiser | SGD (momentum=0.9, weight decay=5e-4) | Adam (lr=0.001) | Faster convergence; no manual LR schedule required |
| LR schedule | Manual step decay | `ReduceLROnPlateau` | Automatic; adapts to training dynamics rather than fixed schedule |
| Training hardware | 2× GTX 580 (3 GB each), 5–6 days | Single CPU/GPU | Accessible reproduction |
| Dataset | ImageNet (1.2M images, 1000 classes) | CIFAR-10 (50K images, 10 classes) | Same architectural principles, tractable scale |
| Data augmentation | Random crops + horizontal flips + colour jitter | Random crops + horizontal flips | Colour jitter omitted; two transforms are sufficient to demonstrate the principle |

The most structurally important preserved choices are: **ReLU activations**, **Dropout (p=0.5) in FC layers**, **overlapping MaxPool (3×3, stride 2)**, and **five stacked convolutional blocks** — the combination that made AlexNet qualitatively different from prior CNN architectures.

---

## References

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). **ImageNet Classification with Deep Convolutional Neural Networks**. *Advances in Neural Information Processing Systems (NeurIPS)*, 25, 1097–1105.
- Krizhevsky, A. (2009). **Learning Multiple Layers of Features from Tiny Images**. *Technical Report, University of Toronto*. *(CIFAR-10 dataset paper)*
- CIFAR datasets: https://www.cs.toronto.edu/~kriz/cifar.html
- Previous project: [LeNet-5 on MNIST (1998)](../LeNet-MNIST-1998/README.md)
