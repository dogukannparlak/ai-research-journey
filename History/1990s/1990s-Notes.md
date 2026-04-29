# AI Architecture: The 1990s - Foundations of Deep Architecture

These notes cover six foundational sources:

| Year | Author(s) | Contribution |
| ---- | --------- | ------------ |
| 1994 | Bengio, Simard, Frasconi | Vanishing/Exploding Gradient - the long-term dependency problem |
| 1995 | LeCun & Bengio | Convolutional Networks - vision, speech, and time series |
| 1995 | Hinton, Dayan, Frey, Neal | Wake-Sleep Algorithm - Helmholtz Machine, unsupervised learning |
| 1997 | Schuster & Paliwal | Bidirectional RNNs (BRNN) |
| 1997 | Hochreiter & Schmidhuber | Long Short-Term Memory (LSTM) |
| 1998 | LeCun, Bottou, Bengio, Haffner | Gradient-Based Learning - LeNet, GTN, document recognition system |

---

## 1) Historical Context: The Maturation of AI Architecture in the 1990s

The backpropagation breakthrough of 1986 ran into two difficult realities:

- In deep networks, the gradient signal **vanishes** or **explodes** as the number of layers increases.
- Fully connected networks are inefficient for images and sequences: the number of parameters becomes astronomical, and there is no translation invariance.

The two major architectural answers of this decade were **CNNs** (local connectivity + weight sharing) and **LSTMs** (memory control through hidden-state gates).

---

## 2) Vanishing/Exploding Gradient - Bengio et al. (1994)

### Core Problem

In recurrent networks (RNNs) and deep networks, the transfer of gradients through time or across layers during backpropagation involves the following product:

```
δ(t) = f'(z(t)) · W · δ(t+1) · f'(z(t+1)) · W · ...
```

When going T steps backward, the gradient has a structure similar to:

```
δ(1) ≈ (W · diag(f'(z)))^T · δ(T+1)
```

Let λ₁ be the largest eigenvalue of this matrix:

- If **λ₁ < 1**: as T grows, the gradient goes **exponentially -> 0** (vanishing gradient).
- If **λ₁ > 1**: as T grows, the gradient goes **exponentially -> ∞** (exploding gradient).

### Technical Details

#### 2.1 Difficulty of Long-Term Dependencies

- If a network's output depends on an input from long ago (t = 1), the gradient of the weight at t = 1 must be carried across T steps.
- For sigmoid: f'(z) ≤ 0.25, so the gradient quickly disappears in long sequences.
- Practical result: RNNs can learn only **short-term dependencies**.

#### 2.2 Three Architectural Approaches

Bengio et al. examine three solution paths:

| Approach | Mechanism | Problem |
| -------- | --------- | ------- |
| Converging to fixed points | Force the system into a stable state | Learns only nearby dependencies |
| Input/output separation | Keep the hidden state constant | Not flexible |
| Special architecture (LSTM) | Control gradient flow with gates | Next solution |

> **Critical takeaway**: Vanishing gradient is not a problem solved only by a better optimization algorithm; it requires a change in the **architecture itself**.

---

## 3) Convolutional Networks - LeCun & Bengio (1995)

### Why Fully Connected Networks Are Not Enough

A fully connected network has three core problems for images or audio:

- **Parameter count**: a 100x100 image with 1000 hidden units -> 10 million weights. Overfitting risk is high.
- **Topological blindness**: If the input order changes, the network does not know; spatial or temporal structure is ignored.
- **Lack of invariance**: If the same character is shifted or rotated slightly, it must be learned again.

### Technical Details

#### 3.1 Three Architectural Ideas of CNNs

**1. Local Receptive Field**

Each hidden unit connects only to a small neighborhood of the input:

```
h(i,j) = f(Σₘ Σₙ w(m,n) · x(i+m, j+n) + b)
```

This allows the neuron to learn local features such as **edges, corners, and textures**.

**2. Weight Sharing / Parameter Tying**

The same receptive-field weights (filter/kernel) are shared across all positions in the image:

```
Feature map:  H(i,j) = f((W * X)(i,j) + b)   [* = convolution]
```

- This operation is mathematically equivalent to **convolution**.
- Multiple filters -> multiple **feature maps**.
- Parameters per filter: kernel_h x kernel_w (for example, 5x5 = 25), independent of position.

**Critical advantage**: Even if there are 60,000 connections, sharing reduces the number of free parameters to only about 1,000.

**3. Subsampling / Pooling**

The feature map is reduced using local averaging or maximum selection:

```
Pool(i,j) = mean or max { H(2i+m, 2j+n), m,n ∈ {0,1} }
```

- The exact location of a feature becomes less important.
- The model gains **robustness** against small translations and distortions.
- As resolution decreases, the number of feature maps increases with depth ("bi-pyramid").

#### 3.2 Layer Ordering

Typical CNN architecture:

```
Input -> [Conv -> Pool] -> [Conv -> Pool] -> ... -> Fully Connected -> Output
```

- Early layers: low-level features such as edges and orientation.
- Middle layers: parts and object components.
- Final layers: high-level abstract representations.

#### 3.3 Variable-Size Input: SDNN

A fixed-size CNN recognizes a single character. But written words or sentences are variable length.

**Space Displacement Neural Network (SDNN)**: applies the CNN by sliding it over the input and produces a sequence of probabilities. This sequence is combined with an HMM (Hidden Markov Model) to make the final decision.

#### 3.4 CNNs in the Time Dimension: TDNN

When a 1D convolutional network is applied to a time series or speech spectrogram, it is called a **Time Delay Neural Network (TDNN)**. With subsampling, it can be used for speech recognition or handwriting recognition.

> **Architectural principle**: LeCun and Bengio clearly state the idea that "feature extractors do not need to be hand-designed - constrain the architecture correctly and let backpropagation learn."

---

## 4) Wake-Sleep Algorithm - Hinton et al. (1995)

### Problem: Unsupervised Representation Learning

Backpropagation requires supervised, labeled data. Can we learn the **data distribution** without labels?

**Helmholtz Machine**: a multilayer stochastic network with both bottom-up recognition connections and top-down generative connections.

### Technical Details

#### 4.1 Network Structure

- Each unit is binary stochastic: sᵥ ∈ {0,1}
- Probability that unit v is active:

```
P(sᵥ = 1) = sigmoid(bᵥ + Σᵤ sᵤ wᵤᵥ)
```

- **Recognition weights**: input -> hidden layers (bottom-up)
- **Generative weights**: hidden layers -> input (top-down)

#### 4.2 Minimum Description Length (MDL) Objective

Goal: find a hidden representation that allows the input to be described with **minimum length**.

Cost of describing the state of unit j:

```
C(sⱼ) = -sⱼ log pⱼ - (1 - sⱼ) log(1 - pⱼ)
```

Total cost:

```
C(τ; d) = C(τ) + C(d|τ)
     = Σ layers C(sⱼ) + Σ input C(sᵢ|τ)
```

This reduces through KL divergence to:

```
C(d) = -log P(d) + KL[ Q(τ|d) || P(τ|d) ]
```

- First term: log-likelihood of the generative model.
- Second term: difference between the recognition distribution and the true posterior.

#### 4.3 Two Phases

**Wake Phase:**

- The input is processed upward through recognition weights -> total representation τ.
- Generative weights are updated to reconstruct the lower layer using the delta rule:

```
Δwₖⱼ = ε · sₖ · (sⱼ - pⱼ)
```

- Only **generative** weights learn.

**Sleep Phase:**

- Samples are generated from the top downward using generative weights -> "fantasy" vectors.
- Recognition weights are updated to recover these fantasies:

```
Δwⱼₖ = ε · sⱼ · (sₖ - qₖ)
```

- Only **recognition** weights learn.

#### 4.4 Architectural Meaning

- **Bidirectional network**: recognition (encoder) + generation (decoder) - a precursor to modern **autoencoder** and **VAE** architectures.
- **Factorial posterior approximation**: units in the hidden layer are assumed conditionally independent - necessary for computational efficiency.
- **Explaining away problem**: Because the true posterior is not factorial, Q(τ|d) ≠ P(τ|d); the wake phase tries to keep the generative model close to Q.

> **Modern connection**: Wake-sleep is an early special case of variational inference (Evidence Lower Bound) and VAEs. The name "Helmholtz Machine" comes from Helmholtz's idea of the perceptual system as a generative model.

---

## 5) Bidirectional RNNs - Schuster & Paliwal (1997)

### Problem

A standard RNN uses only past information (t₀ ... t) when making a prediction at time t. But in many areas such as speech, NLP, and bioinformatics, **future context** also matters.

### Technical Details

#### 5.1 BRNN Architecture

The hidden state of the RNN is split into two independent subsets:

- **Forward states**: processed from t = 1 to T.
- **Backward states**: processed from t = T to 1.

```
Forward:   h→(t) = f(W_x x(t) + W_h h→(t-1) + b)
Backward:  h←(t) = f(W_x x(t) + W_h h←(t+1) + b)
Output:    y(t)  = g(W_out [h→(t); h←(t)] + b_out)
```

- Forward and backward state neurons are **not connected** to each other.
- The output layer combines the states from both directions.

#### 5.2 Training with BPTT

In unfolded form, the BRNN becomes a standard feedforward network; BPTT (Backpropagation Through Time) is applied.

Boundary conditions:

- h→(t=0) = 0.5 (fixed initial state)
- h←(t=T+1) = 0.5 (fixed initial state)

#### 5.3 Experimental Results

Artificial data (asymmetrically dependent time series) and TIMIT phoneme classification:

| Architecture | TIMIT Test Recognition Rate |
| ------------ | --------------------------- |
| MLP-1 (single frame) | 59.67% |
| MLP-3 (3 frames) | 65.69% |
| Forward RNN (1-frame delay) | 65.83% |
| Merged Forward+Backward (MERGE) | 65.28% |
| **BRNN** | **68.53%** |

#### 5.4 BRNN for Dependent Outputs

To compute a full sequence P(c₁, c₂, ..., cₜ | x) instead of a single class prediction:

```
P(c₁...cₜ | x) = P(ct | ct-1, ..., c₁, x) · P(ct-1 | ...)  (forward decomposition)
               = P(c₁ | c₂, ..., cₜ, x) · ...              (backward decomposition)
```

Both decompositions are estimated with separate modified BRNNs and then combined.

> **Modern connection**: BRNN is a direct ancestor of Bi-LSTM in NLP and bidirectional attention mechanisms in Transformers.

---

## 6) Long Short-Term Memory (LSTM) - Hochreiter & Schmidhuber (1997)

### Motivation

The **architectural solution** to the vanishing gradient problem shown in Section 2 (Bengio et al., 1994) is LSTM.

**Main idea**: create a **memory cell (cell state)** where information can be preserved for arbitrary durations, together with **gates** that control gradient flow.

### Technical Details

#### 6.1 LSTM Cell: Components

Each LSTM unit consists of four components:

**Forget Gate**:
Determines what information will be deleted from the cell:

```
fₜ = sigmoid(Wf · [hₜ₋₁, xₜ] + bf)
```

**Input Gate**:
Determines how much new information will be written into the cell:

```
iₜ = sigmoid(Wi · [hₜ₋₁, xₜ] + bi)
c̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)
```

**Cell State Update**:
Part of the old memory is forgotten, and new information is added:

```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ c̃ₜ
```

⊙ = element-wise (Hadamard) product.

**Output Gate**:
Determines which part of the cell state is transferred to the output:

```
oₜ = sigmoid(Wo · [hₜ₋₁, xₜ] + bo)
hₜ = oₜ ⊙ tanh(Cₜ)
```

#### 6.2 Why the Vanishing Gradient Problem Is Solved

Looking at the update of cell state Cₜ:

```
∂Cₜ/∂Cₜ₋₁ = fₜ
```

If the forget gate is approximately 1, the gradient is transferred **fully** from Cₜ to Cₜ₋₁ - it does not vanish.

Comparison:

```
Standard RNN:  ∂hₜ/∂hₜ₋₁ = f'(zₜ) · W  -> risk of exponential decay
LSTM:          ∂Cₜ/∂Cₜ₋₁ = fₜ ∈ (0,1)   -> controlled by the gate
```

#### 6.3 Architectural Properties

- **Constant Error Carousel (CEC)**: The gradient flowing through the cell state can move without vanishing or exploding.
- **Gating mechanism**: Learned gates decide what information is stored, deleted, and when.
- **Hidden state vs. cell state**: h(t) represents short-term output; C(t) represents long-term memory.

#### 6.4 Long-Term Dependencies

The original LSTM paper tests the architecture on:

- **Embedded Reber Grammars**: grammar learning with dependencies more than 100 steps back.
- **Minimal time lags**: the input at t = 1 affects the output at t = 1000.
- **Parity tasks**: parity checking over bits.

While standard RNNs and other algorithms fail, LSTM learns successfully.

> **Modern impact**: LSTM and its variants (GRU - Gated Recurrent Unit) were the dominant architecture for sequence-to-sequence tasks such as language modeling, speech recognition, and machine translation until the mid-2010s. They remained the standard architecture until the Transformer in 2017.

---

## 7) Gradient-Based Learning - LeCun et al. (1998): LeNet and GTN

### Approach: Globally Trained Modular System

The 1998 paper presents a complete document recognition architecture that is a precursor to modern deep learning systems.

**Core claim**: Instead of hand-designed feature extractors and heuristic segmentation, the whole system (feature extraction + classification) can be optimized **end to end** with gradient-based learning.

### Technical Details

#### 7.1 LeNet Architecture

LeNet-5, designed for digit recognition, is the reference architecture of modern CNNs:

```
Input: 32x32 pixels

C1:  6 feature maps, 5x5 kernel  -> 28x28
S2:  Subsampling 2x2             -> 14x14    (6 feature maps)
C3:  16 feature maps, 5x5 kernel -> 10x10
S4:  Subsampling 2x2             -> 5x5      (16 feature maps)
C5:  120 feature maps, 5x5       -> 1x1      (like fully connected)
F6:  84 units, fully connected
Output: 10 classes (digits 0-9)
```

- Total connections: about 340,000
- Free parameters: about 60,000, thanks to weight sharing

#### 7.2 Loss Function: From MSE to Cross-Entropy

For multiclass classification, cross-entropy loss within the **Maximum Likelihood Estimation** framework:

```
L = -Σₖ yₖ · log(ŷₖ)
```

Softmax output:

```
ŷₖ = exp(zₖ) / Σⱼ exp(zⱼ)
```

#### 7.3 Graph Transformer Networks (GTN)

For word-level or document-level training, instead of **heuristic segmentation**, the approach is:

- Slide the character recognizer across all positions -> output sequence (SDNN).
- Feed this output sequence into a graphical model solved by the Viterbi algorithm (directed acyclic graph).
- Compute gradients through the graph and train the **entire system end to end**.

This approach is called a "Trainable Graph Transformer" (GTN), combining HMMs and CNNs within a single differentiable computation framework.

#### 7.4 Performance (MNIST)

| Method | Error Rate |
| ------ | ---------- |
| K-NN (pixels) | 5.00% |
| SVM (Gaussian) | 1.40% |
| Simple FC network | 4.50% |
| LeNet-1 | 1.70% |
| **LeNet-5** | **0.95%** |

LeNet-5 reached state-of-the-art results at the time and was integrated into a commercial system for reading bank checks in the 1990s.

> **Architectural lesson**: LeNet-5 and GTN together provide concrete evidence for the transition from the "feature engineering + modular system" paradigm to the "end-to-end differentiable system" paradigm.

---

## 8) Architectural Synthesis: Core Lessons of the 1990s

### Technical Details

#### 8.1 Comparison Across Architectural Dimensions

| Architecture | Topology | Representation | Success Area | Limitation |
| ------------ | -------- | -------------- | ------------ | ---------- |
| CNN (LeCun) | Layered, local, shared | Hierarchical features | Vision, speech, 1D sequences | No long-term dependency |
| Helmholtz/Wake-Sleep | Bidirectional, stochastic | Generative + recognition | Unsupervised, generative | Factorial approximation limit |
| BRNN | Bidirectional recurrent | Bidirectional temporal | Sequence classification | Requires the whole sentence |
| LSTM | Gated recurrent | Cell state (long-term) | Long sequence modeling | Difficult parallel training |
| LeNet/GTN | CNN + differentiable graph | End-to-end features + decision | Document recognition | Large data was unavailable |

#### 8.2 Lasting Architectural Contributions of This Decade

1. **Weight sharing**: The core way to reduce parameters and add structural inductive bias.
2. **Gating mechanism**: Learnable control over gradient flow and information storage/deletion.
3. **End-to-end learning**: Components are optimized as a system, not separately.
4. **Generative models**: Learning generative models alongside discriminative recognition.
5. **Context integration**: A forward direction is not enough; use the full context, past and future.

#### 8.3 Agenda for the Next Era (Questions Left for the 2000s)

- **Training deep networks**: LeNet had 5 layers; training deeper networks was still problematic.
- **Pretraining vs. end-to-end**: Hinton et al. would address this in 2006 with greedy layer-wise pretraining.
- **Lack of large data**: The 1990s had small datasets like MNIST; the ImageNet explosion in the 2010s would change architectural decisions.
- **Computational power**: GPU-based training was not yet common; AlexNet would change this in 2012.

---

## 8.4 Practical Experiment: LeNet-5 Replication

**Project:** `Deep Learning/CNN/LeNet-MNIST-1998/`

The LeNet-5 architecture from LeCun et al. (1998) was implemented in PyTorch, trained on MNIST, and compared with classical ML methods.

### Experimental Conditions

- Python 3.12, PyTorch 2.11, CPU (no GPU)
- Optimizer: Adam (lr=0.001), 10 epochs, batch=64
- Data: 54,000 train / 6,000 validation / 10,000 test

### Results

| Model | Test Accuracy | Train Time |
| ----- | ------------- | ---------- |
| **LeNet-5** | **98.69%** | ~5 min (CPU) |
| kNN (k=3) | 94.52% | ~0s |
| SVM (LinearSVC) | 91.61% | ~18 min |

Original paper result: LeNet-5 had **0.95% error** (that is, 99.05% accuracy).
Replication: **1.31% error** - an expected deviation for 10 epochs of CPU training.

### Observations

- LeNet-5 outperforms classical methods by 4-7 points.
- The hardest digit is **5** (confused with 3 and 6) - common across all models.
- kNN producing 94.5% with zero training time shows that distance-based methods are strong in pixel space.
- LinearSVC is weak because a linear decision boundary cannot capture the complex morphology of handwriting; an RBF-kernel SVM could reach the paper's result of about 1.4% error.

### Historical Meaning

This experiment confirms the claim of the 1998 paper: **CNNs can learn directly from raw pixel data without hand-designed features** and outperform classical methods such as K-NN and SVM. LeNet-5 is one of the first successful large-scale proofs of the end-to-end differentiable system paradigm.

---

## 9) Critical Points Summary

- **Vanishing gradient**: The f' ≤ 0.25 bound of sigmoid activations drives gradients to zero in deep networks; LSTM solves this with a gating mechanism.
- **CNN triad**: Local receptive fields + weight sharing + pooling -> translation invariance + parameter efficiency.
- **LSTM cell state**: Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ c̃ₜ - allows units to retain information for arbitrary durations.
- **BRNN**: Output prediction uses the entire sequence, not only the past; bidirectional processing.
- **Helmholtz Machine**: Separating generative and recognition weights enables unsupervised representation learning.
- **GTN/LeNet**: Segmentation heuristics can be removed; the whole pipeline can be differentiable.
- **Weight sharing**: Reduces parameter count, improves generalization, and embeds structural knowledge such as translation invariance into the model.
