# AI Architecture: Perceptron-Based Architectural Notes (Pre-1980)

These notes explain the architecture of a "learning system" from a university-level **AI Architecture** perspective, mainly based on Rosenblatt's **Perceptron** approach (1958) and the accompanying formulations from lecture notes.

---

## 1) Historical Background and Motivation

Rosenblatt's initial motivation was to place three fundamental questions about biological intelligence into an architectural framework:

- **Sensing/detection**: How is information perceived from the physical world?
- **Storage/memory**: In what form is information stored in "memory"?
- **Influence on recognition/behavior**: How does stored information guide recognition and behavior?

These questions produce two main approaches:

- **Coded memory**: the idea that a stimulus is represented in memory by a "code" and can later be reconstructed.
- **Connectionist memory**: the idea that information is not stored as an "image", but through changes in **connections/transmission paths**; recognition does not require a separate "matching" process because the new stimulus automatically uses already strengthened paths.

### Technical Details

The McCulloch-Pitts (M-P) logical neuron:

- It is a threshold unit with **fixed parameters**; weights and threshold values are selected by "design".
- Networks are built like "lego": they are pre-designed to implement a desired Boolean function.
- There is no **learning/adaptation** mechanism.

Rosenblatt's core architectural contribution:

- By making parameters such as weights **wᵢ** and threshold/bias **ϑ** **learnable** variables, he shifts the architecture from "design" to "learning".

---

## 2) Perceptron Architecture

The perceptron is designed as a layered and partially randomly connected network: a "learning information processing system".

- **S-units (S-points)**: sensory units / "retina" points
- **A-units (association units)**: association/intermediate representation units
- **R-units (responses)**: output/response units (classification, decision)

In the classical photo-perceptron description, the layers are more detailed:

```
Retina (S)  ->  Projection Area (A₁)  ->  Association Area (Aₙ)  ->  Responses (R)
```

### Technical Details

#### 2.1 Connection Organization

- **S -> A₁**: More "regular/local" connections; the "origin points" set of each A₁ unit is distributed **focally** over the retina (more connections from nearby neighborhoods, exponentially fewer as distance increases). This creates a basis for functions such as **contour detection**.
- **A₁ -> Aₙ**: **Random** connections; Aₙ units receive distributed sources from the A₁ area.
- **Aₙ -> R**: An R unit is fed through a **source-set** over Aₙ.

Architectural meaning of this organization:

- Instead of the input space (retina) going directly to a single output, it is decomposed through **intermediate representations** (A-units).
- Randomness supports the idea of **rough organization + adjustment through learning**, instead of a deterministic design that is "fully genetically wired".

#### 2.2 Feedback and Competition

The perceptron description discusses bidirectional connections between R and A (feedback):

- **Excitatory feedback**: R excites its own source-set (possibly more biologically plausible).
- **Inhibitory feedback** (easier to analyze): R suppresses A units outside its own source-set (inhibits the complement).

In practice, this produces the following architectural behavior:

- **Mutual exclusivity / competition**: When one R becomes active, it suppresses the others; a winner-take-all style decision emerges.
- The system is considered in two phases:
  - **Predominant phase**: some A-units become active, but R has not yet become clear.
  - **Postdominant phase**: one R becomes dominant and competing sets are suppressed.

---

## 3) Mathematical Model of the Perceptron (Single-Neuron Level)

The "two-layer perceptron" form in the lecture notes is the common engineering representation of a threshold neuron:

- Input: **x = (x₁, x₂, ..., xₚ)**
- Weights: **w = (w₁, w₂, ..., wₚ)**
- Threshold/bias: **ϑ**
- Potential (net input):

```
ξ = w₁x₁ + w₂x₂ + ... + wₚxₚ + ϑ
  = Σᵢ wᵢxᵢ + ϑ
```

- Output (step/threshold function):

```
       ⎧ 1,  ξ ≥ 0
y = s(ξ) = ⎨
       ⎩ 0,  otherwise
```

### Technical Details

#### 3.1 Continuous Perceptron

Many problems require **continuous function approximation** rather than classification. For this:

- Output: **y = t(ξ)**, where **t : ℝ -> (0,1)** is differentiable and monotonically increasing.
- Typical transfer function: **sigmoid**

```
           1
t(ξ) = ─────────────
        1 + e^(-α·ξ)
```

- As **α** increases, the sigmoid approaches the step function.

This form is important for understanding the historical line leading to activation functions in modern neural networks (sigmoid/tanh/ReLU, etc.). In modern terms, this points toward **backpropagation**; here the gradient descent perspective is built through a single neuron/single layer.

---

## 4) Learning Algorithm (Perceptron Learning Rule)

A training example:

- **Aₖ = (xᵏ, ŷᵏ)** -> xᵏ is the pattern vector, and ŷᵏ ∈ {0, 1} is the target output.

The goal is to find **w*** and **ϑ** such that:

- For the positive class (ŷ = 1): **w* · x + ϑ > 0**
- For the negative class (ŷ = 0): **w* · x + ϑ < 0**

### Technical Details

#### 4.1 Rosenblatt / Hebbian Update

Potential computation:

```
ξᵏ = Σᵢ wᵢ · xᵢᵏ + ϑ
```

Output:

```
yᵏ = s(ξᵏ)
```

Update with learning rate **λ > 0**:

```
w_new = w_old + λ · (ŷᵏ - yᵏ) · xᵏ

ϑ_new = ϑ_old + λ · (ŷᵏ - yᵏ)
```

Architectural interpretation of this rule:

- When there is a **misclassification** (ŷᵏ ≠ yᵏ), the weight vector is "pushed" toward the target class.
- If the classification is correct, no update is made (error term = 0).

#### 4.2 Short Example: Boolean OR

Using x₀ = 1 for the bias, write **w = (ϑ, w₁, w₂)**.

Training set (OR):

| Example | x₀ | x₁ | x₂ | ŷ |
|---------|----|----|----|----|
| A₁ | 1 | 0 | 0 | 0 |
| A₂ | 1 | 0 | 1 | 1 |
| A₃ | 1 | 1 | 0 | 1 |
| A₄ | 1 | 1 | 1 | 1 |

Important point:

- By updating **w** on misclassified examples, the algorithm can eventually find a linear boundary (hyperplane) that separates OR.
- This is the basis of the perceptron's strength for **linearly separable** problems.

#### 4.3 Convergence Guarantee (Novikoff Theorem)

> If the training data is **linearly separable**, the perceptron learning algorithm is guaranteed to find a solution in a **finite number of steps**.

Architecturally, this clarifies the capacity and the limit of the "single layer + threshold" structure.

---

## 5) Statistical Separability and Architectural Parameters

Rosenblatt's 1958 paper treats the perceptron as a probabilistic model and analyzes performance/learning using several key probabilistic parameters.

- **Pₐ**: the expected fraction of A-units activated by a stimulus.
- **Pᶜ**: the conditional probability of "shared activation" between two stimuli (a similarity measure).

### Technical Details

#### 5.1 Effect of Threshold and Inhibition

Rosenblatt's analysis mathematically confirms the following engineering intuition:

- Increasing the threshold **ϑ**:
  - decreases both **Pₐ** and **Pᶜ**,
  - may decrease **Pᶜ** more sharply -> enables **tighter discrimination**.
- Increasing the proportion of inhibitory connections:
  - sparsifies activation and can reduce unnecessary shared activation.

These kinds of settings resemble modern architectural ideas:

- activation sparsity, regularization, sharper discrimination, and robustness to noise.

#### 5.2 Two Environment Models: Ideal vs. Differentiated Environment

Rosenblatt defines two extreme scenarios:

- **Ideal environment**: stimuli are random; there is no "internal class structure".
  - Learning may exceed chance in some associations.
  - But the basis for **generalization** is weak.
- **Differentiated environment**: there are multiple classes; stimuli from the same class are correlated with each other.
  - The difference between **Pᶜ₁₁** (within-class similarity) and **Pᶜ₁₂** (between-class similarity) makes generalization possible.

Critical inequality, readable as an architectural design criterion:

```
Pᶜ₁₂  <  Pₐ  <  Pᶜ₁₁
```

When this holds:

- Learning/generalization approaches an asymptote above chance.
- As the number of A-units **Nₐ** increases, performance can approach the upper bound.

---

## 6) System Dynamics: α / β / γ Systems

Rosenblatt studies three basic systems through the "value" (**V**) dynamics of A-units:

- **α-system (Uncompensated Gain)**: an active unit gains value and this accumulation is preserved.
- **β-system (Constant Feed)**: a fixed total gain is distributed across activity for each source-set.
- **γ-system (Parasitic Gain)**: active units gain while inactive units in the same source-set lose value -> total value remains balanced.

### Technical Details

| System | Active unit | Inactive unit | Average V |
|--------|-------------|---------------|-----------|
| α | +1 / firing | unchanged | increases |
| β | +K/Nₐᵣ | -K/Nₐ | increases |
| γ | +1 | -Nₐᵣ/Nₐ | **constant** |

Architectural implications:

- **α**: Learning is cumulative, but in the long term "growing values" can amplify random deviations.
- **β**: Because total value grows over time, small statistical differences can be exaggerated -> instability/noise increases.
- **γ**: A resource-limited competitive dynamic; it may behave more stably in distributed systems.

---

## 7) Bivalent Systems and Trial-and-Error Learning

In monovalent systems, active units are always strengthened in a "positive" direction. In bivalent systems:

- Active units can both increase and decrease through **positive reinforcement** and **negative reinforcement**.
- Behaviorally, this is close to a **reward/punishment** analogy.

### Technical Details

Critical architectural points:

- With binary-coded response patterns, different feedback can be applied for each bit depending on its on/off state.
- The disjoint source-set assumption may be critical for bivalent systems to work properly.

Engineering interpretation:

- In an architecture where responses overlap too much (very high shared connectivity), positive and negative signals may "blur" each other.

---

## 8) Continuous Perceptron: Loss Function and Gradient Descent

In the continuous perceptron, the objective may be **approximation** rather than classification.

Objective function per example:

```
Eₖ(w, ϑ) = 1/2 · (yᵏ - ŷᵏ)²
```

Total objective over all P examples:

```
E(w, ϑ) = Σₖ Eₖ(w, ϑ)   [k = 1, 2, ..., P]
```

### Technical Details

#### 8.1 Batch vs. Online

- **Batch learning**: the gradient is accumulated over the whole dataset, then one update is made.
- **Online learning**: an update is made after each example (an early form of stochastic/online learning).

#### 8.2 Update (Steepest Descent / Gradient Descent)

```
w  ←  w - λ · ∂E/∂w

ϑ  ←  ϑ - λ · ∂E/∂ϑ
```

An important observation in the lecture notes:

- Since the transfer function is monotonically increasing, **t'(ξ) > 0**. With simplification in some online updates, the practical update rule can reduce to the **same form** as the classical perceptron rule.

This hints at the derivative-chain idea at the core of modern **backpropagation**, visible here explicitly for a single neuron.

---

## 9) High-Order Perceptron and Computational Capacity

Following the Minsky and Papert line, the basic limitation of the perceptron is:

- The standard perceptron can separate only **linearly separable** classes.
- Example: **XOR** is not linearly separable -> a single threshold plane is not enough.

A high-order perceptron adds nonlinear terms to the potential:

```
ξ = Σᵢ wᵢxᵢ
  + Σᵢ≤ⱼ wᵢⱼ · xᵢxⱼ              (second order - quadratic)
  + Σᵢ≤ⱼ≤ₖ wᵢⱼₖ · xᵢxⱼxₖ         (third order - cubic)
  + ...
  + ϑ
```

### Technical Details

- This expands the feature space: terms such as **xᵢxⱼ** behave like new "features".
- In the theorem statement from the lecture notes, with appropriate parameterization, a high-order threshold neuron can behave like a **universal approximator** under certain conditions.

Architectural result:

- Capacity growth usually increases both the **number of parameters** and the **computational cost**.
- This historical discussion later triggers the importance of the "hidden neuron / hidden layer" idea.

---

## 10) Architectural Summary: Lessons from the Perceptron for AI Architecture

This section translates perceptron-derived ideas into the language of modern "AI Architecture".

- **Layering**: the retina/association/response separation creates a sequential transformation of representations, a representation pipeline.
- **Random connectivity + learning**: rough topology + adjustment through data reduces the design burden.
- **Competition and feedback**: enables decision mechanisms (winner-take-all) and behaviors similar to attention/selective recall.
- **Probabilistic analysis**: explaining performance through parameters such as **Pₐ** and **Pᶜ** provides an early engineering language for architectural tuning.
- **Limits**: relational abstraction and the linear separability limit clarify where the architecture becomes insufficient.

### Technical Details

#### 10.1 Rosenblatt's Six Fundamental Physical Parameters

Rosenblatt connects learning/performance phenomena to a few fundamental physical parameters, readable as architectural design variables:

| Parameter | Description |
|-----------|-------------|
| **x** | Number of excitatory connections per A-unit |
| **y** | Number of inhibitory connections |
| **ϑ** | A-unit threshold |
| **ω** | Ratio of how many R-units an A-unit connects to (connection density) |
| **Nₐ** | Number of A-units (capacity) |
| **Nᵣ** | Number of R-units (number of classes/outputs) |

Value of this approach:

- Instead of only "fitting the behavior curve", it aims to predict behavior from measurable/concrete parameters.

#### 10.2 Modern Bridge (Concept Map)

Conceptual continuity from the perceptron to modern architectures:

```
Step/threshold             ->  activations such as sigmoid / tanh / ReLU
Hebbian / perceptron update ->  gradient descent / backpropagation family
Random association layers   ->  feature learning / representation learning
Binary response coding      ->  representation efficiency and capacity
```

---

## 11) "Critical Points" Summary for Exam/Project

- **Linearly separable**: the basic capacity limit of the standard perceptron; examples such as XOR are diagnostic.
- **Convergence guarantee**: for linearly separable data, the perceptron learning algorithm finds a solution in finite steps (Novikoff).
- **Architectural parameters**: threshold and inhibition determine the discrimination/noise balance through activation density and similarity correlation.
- **Environment structure for generalization**: an ideal environment (random) does not support generalization; a differentiated environment (within-class correlation) makes generalization possible.
- **Feedback/competition**: stabilizing the decision in the postdominant phase; architecturally, it provides a selection mechanism.
- **Continuous neuron**: the transfer function + loss + gradient descent framework for function approximation connects to the mathematical backbone of modern learning.
