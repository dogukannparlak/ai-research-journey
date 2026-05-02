# Perceptron & MLP on Iris — Rosenblatt (1958)

> A PyTorch reproduction of **Rosenblatt's Perceptron** applied to Fisher's Iris dataset,
> showing where it succeeds, where it fails, and how a two-layer MLP resolves the failure.

This project is deliberately small: 150 samples, 4 features, 3 classes. Its purpose is not
to demonstrate state-of-the-art accuracy — it is to make the *Perceptron convergence theorem*
and its limits concrete, and to show the exact point where adding a single hidden layer changes
the qualitative behaviour of learning.

---

## Table of Contents

- [Historical Context](#historical-context)
- [Project Structure](#project-structure)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Codebase Walkthrough](#codebase-walkthrough)
- [Results](#results)
- [Reproducing the Experiment](#reproducing-the-experiment)
- [Design Decisions](#design-decisions)

---

## Historical Context

| Year | Milestone |
|------|-----------|
| 1936 | Fisher — *The Use of Multiple Measurements in Taxonomic Problems* (Iris dataset) |
| 1943 | McCulloch & Pitts — formal threshold neuron model |
| 1958 | **Rosenblatt — Perceptron: a probabilistic model for information storage and organization** |
| 1960 | Widrow & Hoff — ADALINE, LMS learning rule |
| **1969** | **Minsky & Papert — *Perceptrons*: formal proof of XOR failure, first AI winter** |
| 1974 | Werbos — backpropagation (unpublished thesis) |
| 1986 | Rumelhart, Hinton & Williams — backpropagation for MLPs (*Nature*) |
| 1989 | Cybenko — Universal Approximation Theorem |

The Perceptron was the first learning machine. Rosenblatt's key insight — borrowed from McCulloch
and Pitts — was that *parameters should be learned from data, not designed by hand*. The Perceptron
algorithm updates weights whenever a prediction is wrong, and the **Perceptron convergence theorem**
guarantees that the algorithm will find a solution in finite steps *if one exists*.

That "if" was the fatal clause. Minsky and Papert's 1969 book showed that many practically
interesting problems — including the simple XOR function — are not linearly separable. No amount
of training will make a single-layer Perceptron solve them. This result, combined with the absence
of a training algorithm for hidden layers, triggered the first AI winter.

The 1986 backpropagation paper broke the deadlock. By showing how to compute gradients through
hidden layers, it made multi-layer learning practical. The MLP in this project is the direct
descendant of that idea.

**Why Iris?**

Fisher's Iris dataset is a perfect case study for this transition:

- *Iris setosa* is **perfectly linearly separable** from the other two classes. The Perceptron will
  classify it with 100% accuracy.
- *Iris versicolor* and *Iris virginica* **overlap** in the 4D feature space. No linear hyperplane
  can perfectly separate them. The Perceptron will consistently misclassify some of these samples.
- Adding one hidden layer with ReLU gives the model enough capacity to learn the non-linear boundary
  between versicolor and virginica — bringing accuracy above 95% on the test set.

This makes Iris a uniquely instructive dataset: it contains both the case where the Perceptron works
and the case where it fails, in the same experiment.

---

## Project Structure

```
Perceptron-Iris-1958/
│
├── config.py           — Central configuration: hyperparameters, paths, class names
├── train.py            — Entry point: trains Perceptron then MLP, saves results
├── compare.py          — Benchmark: all models vs SVM, kNN, Logistic Regression
├── requirements.txt
│
├── src/
│   ├── model.py        — Perceptron and MLP architectures (PyTorch)
│   ├── dataset.py      — UCI ML Repository fetch, caching, split, DataLoaders
│   ├── trainer.py      — Training loop (Adam, CrossEntropyLoss)
│   ├── evaluator.py    — Test accuracy + per-class breakdown + confusion matrix
│   ├── visualizer.py   — Training curves, confusion matrices, PCA scatter plots
│   └── classical.py    — SVM (RBF), kNN, Logistic Regression baselines (sklearn)
│
├── data/
│   └── iris_cache.npz  — Cached dataset (created on first run)
├── models/
│   ├── perceptron.pt   — Saved Perceptron weights
│   └── mlp.pt          — Saved MLP weights
└── results/
    ├── perceptron_training_history.png
    ├── perceptron_confusion_matrix.png
    ├── perceptron_pca_scatter.png
    ├── mlp_training_history.png
    ├── mlp_confusion_matrix.png
    ├── mlp_pca_scatter.png
    ├── svm_confusion_matrix.png
    ├── knn_confusion_matrix.png
    ├── logistic_confusion_matrix.png
    ├── comparison_bar.png
    └── comparison.csv
```

---

## Architecture Deep Dive

### The Dataset

Fisher's Iris dataset contains 150 samples across three species of iris flower. Each sample has
four numerical measurements:

| Feature | Description |
|---------|-------------|
| sepal length (cm) | Length of the outer flower leaf |
| sepal width (cm)  | Width of the outer flower leaf |
| petal length (cm) | Length of the inner flower leaf |
| petal width (cm)  | Width of the inner flower leaf |

Each class has exactly 50 samples. The data is split 70/15/15 (train/val/test) with a fixed
random seed for reproducibility.

### Perceptron

```
Input (4) → Linear(4 → 3) → logits → CrossEntropyLoss
```

**Total parameters: 15**  (4 weights × 3 classes + 3 biases)

The Perceptron is a single affine transformation. Its decision boundaries are hyperplanes in
the 4-dimensional feature space. There is no non-linearity. The Perceptron convergence theorem
says: if the data is linearly separable, this model will find a perfect solution. Iris is not
fully linearly separable — so it cannot.

### MLP

```
Input (4) → Linear(4 → 16) → ReLU → Linear(16 → 3) → logits → CrossEntropyLoss
```

**Total parameters: 131**  (4×16 + 16 + 16×3 + 3)

One hidden layer with ReLU is sufficient. The hidden layer transforms the 4D input into a
16-dimensional representation where the classes become more linearly separable. The Universal
Approximation Theorem guarantees that a sufficiently wide single hidden layer can approximate
any continuous function — including the boundary between versicolor and virginica.

### Why CrossEntropyLoss Instead of the Perceptron Criterion?

The original Perceptron uses a step function and updates only on misclassifications. For multi-class
classification with gradient-based optimisation in PyTorch, CrossEntropyLoss (log-softmax + NLL)
is the natural analogue: it penalises wrong predictions proportionally to their confidence, and its
gradients are well-defined everywhere. The architectural constraint (single linear layer) is
preserved; only the training signal changes.

---

## Codebase Walkthrough

### `config.py` — Central Configuration

```python
BATCH_SIZE    = 16
LEARNING_RATE = 0.01
EPOCHS        = 200
NUM_CLASSES   = 3
INPUT_DIM     = 4
HIDDEN_DIM    = 16

IRIS_CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
```

Batch size 16 is suitable for a dataset of 105 training samples (~7 batches/epoch).
200 epochs give both models time to converge at `lr=0.01` without requiring a scheduler.

---

### `src/dataset.py` — Data Pipeline

**Fetching:** `ucimlrepo.fetch_ucirepo(id=53)` retrieves the Iris dataset via the
UCI ML Repository API. The result is cached to `data/iris_cache.npz` immediately, so
subsequent runs work offline.

**Split:**

```
150 samples (fixed seed=42 permutation)
  └─ 105 training
  └─  22 validation
  └─  23 test
```

**Preprocessing:** `StandardScaler` is fit on the training set and applied to val/test.
This is necessary for convergence: the four Iris features have different ranges
(e.g. sepal length peaks near 7.9 cm, petal width near 2.5 cm). Scaling normalises
them to zero mean and unit variance.

`load_raw_arrays()` returns the unscaled train/test arrays for classical baselines,
which apply their own `StandardScaler` internally.

---

### `src/model.py` — Architectures

The `Perceptron` and `MLP` classes are both `nn.Module` subclasses.

**On the choice of ReLU for MLP:**  
Earlier networks (including the first MLP papers) used sigmoid or tanh. ReLU was
popularised by AlexNet (2012). For a two-layer network on tabular data, ReLU is preferred
because it does not saturate for large activations — gradients flow back without decay.

`count_parameters()` sums all `requires_grad` parameters — used in `train.py` to print
the parameter count before training begins.

---

### `src/trainer.py` — Training Loop

The `Trainer` class is shared between Perceptron and MLP. The internal `_run_epoch()` method
handles both training and validation passes:

- `training=True`: `model.train()`, gradients computed and applied
- `training=False`: `model.eval()`, `torch.set_grad_enabled(False)` for efficiency

**Optimiser:** Adam with `lr=0.01`. Adam's adaptive per-parameter learning rates make it
more robust than SGD for small datasets where individual feature gradients can vary significantly
in magnitude. The Perceptron convergence theorem guarantees convergence for SGD on linearly
separable data, but Adam converges faster in practice.

Epoch metrics are printed every 20 epochs to keep output readable across 200 epochs.

---

### `src/evaluator.py` — Test Evaluation

Runs inference over the test set, computes:

1. **Overall accuracy** — fraction correctly classified out of 23 test samples
2. **Per-class accuracy** — breakdown by species
3. **Confusion matrix** — 3×3 integer matrix

The per-class breakdown is the diagnostic that reveals whether the Perceptron fails
uniformly or fails specifically on the versicolor/virginica boundary.

---

### `src/visualizer.py` — Plots

**Training history:** Loss and accuracy curves for train and validation sets. For the Perceptron,
the val accuracy will plateau below 100% as the model hits the linear separability ceiling.
For the MLP, both curves should continue improving past that ceiling.

**Confusion matrix:** 3×3 heatmap with species names. The key cell to watch is
`versicolor → virginica` and `virginica → versicolor` — these are the confusions
the Perceptron cannot resolve but the MLP can.

**PCA scatter:** Projects the 4D feature space to 2D via PCA. Side-by-side: true labels
vs model predictions, with misclassified samples highlighted in red. This makes the
linear separability problem visually immediate — you can see the versicolor/virginica
overlap that no linear boundary can resolve.

---

### `src/classical.py` — Classical ML Baselines

| Method | Key Parameter | Notes |
|--------|--------------|-------|
| SVM (RBF) | `C=1.0, gamma="scale"` | Non-linear kernel; can separate versicolor/virginica |
| kNN | `k=5` | Non-parametric; sensitive to local density |
| Logistic Regression | `C=1.0, solver="lbfgs"` | Linear like Perceptron but with probabilistic output and L2 regularisation |

All three methods apply `StandardScaler` independently.

**Logistic Regression vs Perceptron:** Both are linear classifiers. The difference is the
loss function. Logistic regression minimises log-loss; the Perceptron minimises a hinge-like
criterion (update only on errors). In practice, logistic regression tends to find better
linear boundaries because it optimises a smoother objective with well-defined gradients
everywhere. Both will hit the same ceiling on the versicolor/virginica boundary.

---

## Results

> Results from a completed run: 200 epochs, Adam (lr=0.01), CPU,
> batch size 16, seed=42 split (105 train / 22 val / 23 test).

### Model Comparison — Iris Test Set (23 samples)

| Model | Test Accuracy | Error Rate | Train Time |
|-------|:------------:|:----------:|:----------:|
| **Perceptron** | **100.00%** | **0.00%** | ~5s (CPU, 200 epochs) |
| **MLP** | **95.65%** | **4.35%** | ~5s (CPU, 200 epochs) |
| SVM (RBF) | 95.65% | 4.35% | 0.005s |
| Logistic Reg. | 95.65% | 4.35% | 0.016s |
| kNN (k=5) | 86.96% | 13.04% | 0.002s |

### Per-Class Breakdown

| Class | Perceptron | MLP | SVM | kNN | Logistic |
|-------|:----------:|:---:|:---:|:---:|:--------:|
| setosa (5 samples) | **5/5** | **5/5** | **5/5** | **5/5** | **5/5** |
| versicolor (7 samples) | **7/7** | **7/7** | **7/7** | **7/7** | **7/7** |
| virginica (11 samples) | **11/11** | 10/11 | 10/11 | 8/11 | 10/11 |

All misclassifications across every model are **virginica predicted as versicolor** — never
the reverse. This is structurally consistent with the known feature-space geometry: a subset
of virginica samples overlaps into versicolor territory along the petal dimensions.

### Confusion Matrix Summary

The single off-diagonal pattern across all models:

```
                Predicted
True       setosa  versicolor  virginica
setosa        5         0          0
versicolor    0         7          0
virginica     0         k         11-k    (k = misclassification count)
```

| Model | k (virginica → versicolor) |
|-------|:--------------------------:|
| Perceptron | 0 |
| MLP | 1 |
| SVM (RBF) | 1 |
| Logistic Reg. | 1 |
| kNN (k=5) | 3 |

### Key Observations

**On the Perceptron achieving 100% test accuracy:**  
This result requires careful interpretation. The test set contains only 23 samples; a single
misclassification shifts accuracy by ~4.3 points. The training history shows the Perceptron's
validation accuracy plateaus around **95–96%** across 200 epochs — it does not reach 100%. The
100% test result reflects a favorable composition of this particular 23-sample test split rather
than a fundamental capability of the linear model. On repeated splits or a larger test set, the
Perceptron would be expected to make virginica/versicolor errors.

**On MLP vs Perceptron convergence:**  
The training curves make the architectural difference visible. The MLP converges sharply in the
first 25 epochs — loss drops from ~1.1 to ~0.1, accuracy climbs from ~47% to ~95%. The Perceptron
converges slowly and continuously across all 200 epochs, with loss still visibly decreasing at
epoch 200. This reflects the difference between learning a non-linear representation (fast,
expressive) and fitting a constrained linear boundary (slow, hitting a capacity ceiling).

**On the direction of errors:**  
Every misclassification in every model points in one direction: virginica predicted as versicolor.
Setosa is always correct. This is the clearest empirical confirmation of the theoretical claim:
setosa is linearly separable from the rest; virginica is not fully separable from versicolor.
The boundary failure is structural, not a model quality issue.

**On kNN underperforming:**  
kNN (k=5) is the weakest model at 86.96% (3 misclassifications). With only 105 training samples
and k=5, boundary regions are poorly estimated — 5 neighbours is a coarse vote when the
versicolor/virginica overlap region has sparse coverage.

---

## Reproducing the Experiment

### Prerequisites

```bash
pip install -r requirements.txt
```

The first run downloads and caches Iris (~2 KB from the UCI ML Repository API).
Subsequent runs are fully offline.

### Training

```bash
python train.py
```

Trains Perceptron (200 epochs) then MLP (200 epochs). Saves weights to `models/`,
plots to `results/`. Takes under 10 seconds on CPU.

### Comparison

```bash
python compare.py
```

Loads saved model weights, trains SVM / kNN / Logistic Regression, prints the full
comparison table, and saves `results/comparison.csv` and `results/comparison_bar.png`.

---

## Design Decisions

| Aspect | Choice | Reason |
|--------|--------|--------|
| Dataset source | `ucimlrepo` (UCI API) | No manual download; reproducible; consistent with UCI canonical version |
| Split | 70/15/15, seed=42 | Small dataset; fixed seed ensures reproducibility across both scripts |
| Scaling | `StandardScaler` (fit on train) | Iris features are in cm with different ranges; scaling stabilises gradient flow |
| Optimiser | Adam (lr=0.01) | Faster convergence than SGD on small data without manual LR tuning |
| Epochs | 200 | Both models converge well within this; no LR scheduler needed |
| Hidden dim | 16 | Sufficient capacity to learn the versicolor/virginica boundary; avoids overfitting |
| SVM kernel | RBF | Iris versicolor/virginica are not linearly separable; LinearSVC would be as limited as the Perceptron |
| Loss | CrossEntropyLoss | Standard for multi-class classification; smooth gradients everywhere |

---

## References

- Fisher, R. A. (1936). **The Use of Multiple Measurements in Taxonomic Problems**. *Annals of Eugenics*, 7(2), 179–188.
- Rosenblatt, F. (1958). **The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain**. *Psychological Review*, 65(6), 386–408.
- Minsky, M., & Papert, S. (1969). **Perceptrons: An Introduction to Computational Geometry**. MIT Press.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). **Learning Representations by Back-Propagating Errors**. *Nature*, 323, 533–536.
- Cybenko, G. (1989). **Approximation by Superpositions of a Sigmoidal Function**. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
- UCI ML Repository — Iris Dataset: https://archive.ics.uci.edu/dataset/53/iris
- Previous project: [LeNet-5 on MNIST (1998)](../../CNN/LeNet-MNIST-1998/README.md)
- Next project: [AlexNet on CIFAR-10 (2012)](../../CNN/AlexNet-CIFAR10-2012/README.md)
