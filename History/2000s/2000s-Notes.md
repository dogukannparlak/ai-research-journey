# AI Architecture: The 2000s - Deep Representations and Pretraining

These notes cover seven foundational sources:

| Year | Author(s) | Contribution |
| ---- | --------- | ------------ |
| 2003 | Bengio, Ducharme, Vincent, Jauvin | Neural Probabilistic Language Model - word embeddings, distributed LM |
| 2006 | Hinton, Osindero, Teh | Deep Belief Nets - RBM stack, greedy pretraining, up-down fine-tuning |
| 2006 | Hinton, Salakhutdinov | Deep Autoencoder - RBM pretraining for nonlinear dimensionality reduction |
| 2007 | Bengio, Lamblin, Popovici, Larochelle | Greedy layer-wise training analysis - continuous inputs, partial supervision |
| 2008 | Vincent, Larochelle, Bengio, Manzagol | Denoising Autoencoders - robust representations, stacked SdA |
| 2009 | Lee, Grosse, Ranganath, Ng | Convolutional Deep Belief Networks - CRBM, probabilistic max-pooling |
| 2009 | Jarrett, Kavukcuoglu, Ranzato, LeCun | Multi-stage recognition - rectification + local normalization dominate |

---

## 1) Historical Context: From Shallow Success to Deep Pretraining

The 1990s established two pillars of modern architecture: **CNNs** (local connectivity, weight sharing) and **LSTMs** (gated recurrence for long dependencies). Yet **training deep feedforward stacks** with backpropagation from random initialization remained unreliable: gradients vanished or exploded, and optimization often stalled in poor basins.

The 2000s reframed the problem:

- **Curse of dimensionality in discrete modeling** (language): n-grams scale combinatorially in context length; similarity between words is not exploited.
- **Unsupervised pretraining as initialization**: learn layer-wise generative structure first, then fine-tune discriminatively or for reconstruction.
- **Bridging generative and discriminative worlds**: RBMs, autoencoders, and their convolutional variants became **feature-learning modules** rather than ends in themselves.

By the end of the decade, the ingredients for the 2010s explosion were visible: **word vectors**, **layer-wise pretraining**, **denoising objectives**, **convolutional generative layers**, and empirical evidence that **architecture (rectification, normalization, depth)** could matter as much as learned filters.

---

## 2) Neural Probabilistic Language Model - Bengio et al. (2003)

### Core Architectural Problem

Statistical language modeling requires **P(w_t | w_1, ..., w_{t-1})**. Classic n-grams back off to shorter contexts and **do not share statistical strength** between similar words: seeing "The cat is walking in the bedroom" should raise the probability of analogous sentences with "dog", "room", etc.

### Technical Details

#### 2.1 Distributed Word Representation

Each word i in vocabulary V maps to a real vector **C(i) in R^m** (embedding row in matrix C of size |V| x m). The model learns **C** jointly with the predictor.

#### 2.2 Decomposition of the Conditional

```
P(w_t | context) = g( w_t ; C(w_{t-n+1}), ..., C(w_{t-1}) )
```

where g is implemented as a neural network (e.g. one tanh hidden layer + softmax over vocabulary).

#### 2.3 Feedforward MLP Form (paper's main architecture)

Concatenate context embeddings into x:

```
x = ( C(w_{t-1}), C(w_{t-2}), ..., C(w_{t-n+1}) )
```

Hidden pre-activation and logits:

```
y = b + W x + U tanh(d + H x)
```

Softmax:

```
P(w_t = i | ...) = exp(y_i) / sum_j exp(y_j)
```

Optional **direct connections** W from x to logits (linear shortcut); weight decay regularizes H, U, W, C.

#### 2.4 Training and Scale

- Objective: penalized log-likelihood (weight decay on network weights and C).
- **Computational bottleneck**: full softmax over |V| at every step - motivates later hierarchical softmax, sampling, and GPU pipelines.
- **Parallel training**: parameter-parallel softmax blocks; MPI Allreduce for normalization and gradient aggregation.

#### 2.5 Mixture with n-grams

Interpolating neural LM with smoothed trigram **reduces perplexity** further - the two models err in different places.

#### 2.6 Architectural Implications

- **Parameter sharing across time positions** for the same word type - precursor to recurrent and transformer token embeddings.
- **Smooth generalization** over sentence space via continuous word features - foundation for Word2Vec, GloVe, and modern LMs.
- **Longer effective context** than back-off n-grams on the same data - shifts LM design from table lookup to **learned composition**.

> **Modern connection**: This paper is the direct ancestor of **neural language modeling** and **static/dynamic word embeddings**; the softmax bottleneck foreshadows sampled softmax and subword tokenization at scale.

---

## 3) Deep Belief Nets and Deep Autoencoders - Hinton et al. (2006)

This section covers two companion breakthroughs from the same research line: **greedy RBM-based pretraining** for deep belief nets (Neural Computation), and **stack-unfold-finetune** for **deep autoencoders** (Science).

### Part A: Deep Belief Nets - Hinton, Osindero, Teh

#### Core Idea

**Explaining away** makes posterior inference in deep directed nets hard. **Complementary priors** (constructed via tied weights in an infinite directed expansion) yield **factorial posteriors** layer-wise, linking deep directed learning to **undirected RBMs** and enabling **efficient layer-wise training**.

#### Technical Details

##### 3.1 Logistic Belief Net Unit

```
P(s_i = 1) = sigmoid( b_i + sum_j s_j w_ij )
```

##### 3.2 Restricted Boltzmann Machine (RBM)

Energy (binary-binary case):

```
E(v, h) = - sum_{i,j} v_i w_ij h_j - sum_i b_i v_i - sum_j c_j h_j
```

**Block Gibbs sampling**: update all h given v, then all v given h - parallels alternating layers in the infinite tied-weight directed net.

##### 3.3 Contrastive Divergence (CD)

Approximate maximum likelihood by running the chain **k** steps (often k=1):

```
d(log p(v))/dw_ij approx <v_i h_j>_data - <v_i h_j>_recon
```

CD minimizes a difference of KL divergences; it is fast but not exact ML.

##### 3.4 Greedy Layer-Wise Construction

1. Train bottom RBM on data.
2. Use inferred hidden activations as **data for the next RBM**.
3. Compose into a **hybrid**: top two layers undirected (associative memory); lower layers directed generative + recognition weights (initially tied).

**Variational bound** (Neal & Hinton): adding a layer, if wide enough and initialized correctly, **improves** a lower bound on log p(v) - motivates greedy stacking.

##### 3.5 Up-Down (Contrastive Wake-Sleep) Fine-Tuning

After greedy pretraining:

- **Up-pass**: recognition weights infer hidden states; update **generative** directed weights and top RBM parameters.
- **Down-pass**: short runs of Gibbs in top associative memory, then generative pass; update **recognition** weights.

This **contrastive** variant mitigates mode-averaging and poor sleep-phase fantasies of classical wake-sleep.

##### 3.6 MNIST Generative-Classifier Results

Architecture (permutation-invariant): **784 -> 500 -> 500 <-> 2000 <-> 10** (labels as softmax group in top RBM).

- Greedy RBM pretraining per layer, then **300 epochs** up-down.
- Test error **1.25%** - competitive with strong SVMs and backprop MLPs without conv structure.

##### 3.7 Architectural Implications (DBN)

- **Unsupervised pretraining** supplies a good parameter basin for deep generative models.
- **Top-level associative memory** enables sampling and **interpretability** via generation.
- **Separation of recognition vs generative weights** after untying - step toward inference networks in later VAE/GAN era.

### Part B: Reducing Dimensionality - Hinton & Salakhutdinov (Science 2006)

#### Core Problem

**Nonlinear autoencoders** with many hidden layers are hard to train from random init: large weights -> poor minima; small weights -> tiny early-layer gradients.

#### Technical Details

##### 3.8 Autoencoder Structure

**Encoder** maps input to code; **decoder** reconstructs. Fine-tuning minimizes reconstruction error (MSE or cross-entropy for [0,1] inputs) via backprop **through both** halves.

##### 3.9 Pretraining Protocol

1. Train a stack of RBMs layer by layer (hidden activations of level L become visible data for level L+1).
2. **Unfold** the stack into a symmetric deep autoencoder (initially tied encoder/decoder weights).
3. **Fine-tune** end-to-end with backprop using deterministic real-valued propagations (probabilities instead of stochastic bits, except where noted).

##### 3.10 Continuous Visible Units

Gaussian visible units with fixed variance: sampling mean **b_i + sum_j h_j w_ij** for v_i | h.

Top RBM may use **linear/Gaussian hidden states** for the code layer to match PCA-like codes.

##### 3.11 Empirical Highlights

- **Curves** synthetic data (known 6D intrinsic structure): deep autoencoder beats PCA and logistic PCA at 6-D code.
- **MNIST** 784-1000-500-250-30: much better reconstructions than PCA; 2-D codes separate classes nonlinearly.
- **Documents** (Reuters): low-dimensional codes improve retrieval vs LSA.
- **Supervised**: pretraining + backprop on 784-500-500-2000-10 achieves **1.2%** MNIST error (reported in supplement reference path).

##### 3.12 Architectural Lesson (Autoencoder)

**Pretraining is an initialization strategy** that makes deep nonlinear dimensionality reduction **practical** - the same principle transfers to supervised deep nets.

---

## 4) Greedy Layer-Wise Training of Deep Networks - Bengio et al. (2007)

### Motivation

**Circuit complexity** intuitions: deep compositions can represent highly varying functions far more efficiently than shallow kernels or single-hidden-layer nets. The missing piece was **how to optimize** deep stacks.

### Technical Details

#### 4.1 Deep Belief Net as Layered Bernoulli Causal Model

```
P(x, g1, ..., gL) = P(x|g1) P(g1|g2) ... P(g_{L-2}|g_{L-1}) P(g_{L-1}, gL)
```

Top **P(g_{L-1}, gL)** is an RBM; each conditional is factorial logistic.

#### 4.2 RBM Energy and CD Training

```
P(v,h) proportional to exp( -energy(v,h) ),  energy(v,h) = -h' W v - b'v - c'h
```

CD-k uses short Gibbs chains for gradient estimates.

#### 4.3 Greedy Procedure (summary)

Train RBM on g^{l-1}; sample or mean-field propagate to form empirical distribution for layer l; stack. Approximate posteriors Q(g^l | g^{l-1}) from RBMs are used even after the generative interpretation shifts - pragmatic inference.

#### 4.4 Supervised Fine-Tuning

After unsupervised pretraining, append classifier; optimize task loss (cross-entropy / MSE) by gradient descent, often with **larger learning rate** on supervised head than during CD.

#### 4.5 Continuous Inputs for RBMs

- **Binomial trick**: treat scaled continuous inputs as probabilities for Bernoulli visible units - works for pixels, weak for arbitrary reals.
- **Gaussian visible RBMs**: quadratic terms in energy; means depend on h.
- **Truncated exponential / exponential** units: sampling changes; CD weight updates structurally similar.

**Partially supervised first layer**: when p(x) is **uncooperative** for predicting y (e.g. finance), mix unsupervised CD updates with supervised gradient through a temporary output layer - large gains on such tasks.

#### 4.6 MNIST Experiments (key comparisons)

| Method | Test error (typical best) |
| ------ | ------------------------- |
| DBN unsupervised pretraining | ~1.2% |
| Deep net, stacked **autoencoder** pretraining | ~1.4% |
| Deep net, **supervised greedy** layer-wise pretraining | ~2.0% |
| Deep net, **no** pretraining | ~2.4% |
| Shallow net, no pretraining | ~1.9% |

**Interpretation**: unsupervised pretraining improves **optimization** and yields **representations** that generalize when the top hidden layer is constrained small (experiment forcing top layer to 20 units: no pretraining fails badly on training error too).

#### 4.7 Architectural Takeaway

**Unsupervised pretraining is not magic** - it is a **regularizer and initializer** that aligns hidden layers with data manifold before expensive supervised reshaping.

---

## 5) Denoising Autoencoders - Vincent et al. (2008)

### Core Principle

A good representation should be **robust to partial corruption**: high-dimensional inputs lie near low-dimensional structure; corrupted points should map back to **clean** originals.

### Technical Details

#### 5.1 Basic Autoencoder

Encoder f_theta, decoder g_theta'; minimize reconstruction loss L(x, g(f(x))) - risk of **identity** if capacity too high.

#### 5.2 Corruption Process

Sample corrupted input:

```
x_tilde ~ q_D(x_tilde | x)
```

Example: **masking noise** - zero out random fraction nu of components.

#### 5.3 Denoising Objective

```
min E_{q0(x) q_D(x_tilde|x)} [ L( x, g_theta'( f_theta(x_tilde) ) ) ]
```

with L often **cross-entropy** for Bernoulli/probability inputs.

#### 5.4 Stacked Denoising Autoencoder (SdA)

Train layer 1 SdA; use **clean** layer-1 outputs as input to train layer 2; ...; then supervised fine-tune entire net.

#### 5.5 Perspectives

- **Manifold**: mapping off-manifold corruptions back onto manifold.
- **Generative bound**: related to maximizing a variational lower bound under a specific graphical model (paper Section 4.2).
- **Information**: denoising relates to capturing **mutual information** between code and uncorrupted x.

#### 5.6 Benchmark Results (Larochelle et al. suite)

**SdA-3** often matches or beats **DBN-3** and **SVMs** across MNIST variants; best nu task-dependent (10-40%). **Overcomplete** first layers become usable - denoising prevents trivial identity.

#### 5.7 Architectural Implications

- **Explicit criterion** for "good" unsupervised features beyond CD stability.
- **Bridge** between denoising in signal processing and deep initialization.
- Foreshadows **dropout** (different mechanism) and **mask-based self-supervised** learning.

---

## 6) Convolutional Deep Belief Networks - Lee et al. (2009)

### Problem

Fully connected DBNs **do not scale** to full images and ignore **translation structure**.

### Technical Details

#### 6.1 Convolutional RBM (CRBM)

Visible image V (N_V x N_V); K groups of hidden feature maps H^k (N_H x N_H); shared filter W^k (N_W x N_W).

Energy (schematic):

```
E(v,h) = - sum_k sum_{i,j} h^k_{ij} ( tilde(W^k) * v )_{ij} - biases
```

Conditionals:

```
P(h^k_{ij}=1 | v) = sigmoid( (tilde(W^k) * v)_{ij} + b_k )
P(v_{ij}=1 | h) = sigmoid( (sum_k W^k * h^k)_{ij} + c )
```

Gaussian visible units for natural image intensities.

#### 6.2 Probabilistic Max-Pooling

Partition each hidden map into non-overlapping C x C blocks. **At most one** detector in a block may be on; **pooling unit** indicates any-on. Sampling is **multinomial** within block - generative counterpart to deterministic max-pool in CNNs.

#### 6.3 Sparsity Regularization

Penalty encouraging low mean activation rho per unit - needed to learn oriented edges instead of trivial global solutions.

#### 6.4 Stacked CDBN

Stack max-pooling CRBMs; greedy layer-wise training; **mean-field** inference often used (e.g. 5 iterations) for speed.

#### 6.5 Hierarchical Inference

Undirected inter-layer edges allow **bottom-up and top-down** signals to combine in block Gibbs - useful for occlusion filling experiments.

#### 6.6 Results Sketch

- Natural images: layer-1 Gabor-like edges; layer-2 corners/contours.
- **Caltech-101** with self-taught natural-image pretraining: strong accuracy (e.g. **65.4%** with 30 training images/class in paper's setting).
- **MNIST**: **0.82%** error with full training set using CDBN features + SVM.

#### 6.7 Architectural Implications

- Unsupervised **convolutional generative** hierarchy at realistic resolutions.
- **Probabilistic pooling** links generative training to CNN inductive biases.
- Sets stage for **convolutional autoencoders**, **deconvnets**, and **variational** conv models.

---

## 7) Multi-Stage Architecture for Object Recognition - Jarrett et al. (2009)

### Research Questions

1. Which **nonlinearities** after filter banks matter?
2. Does **filter learning** beat random/hardwired if architecture is fixed?
3. Is **two-stage** feature extraction better than one?

### Technical Details

#### 7.1 Stage Modules

- **FCSG**: Convolution + tanh/sigmoid + per-feature gain.
- **R_abs**: Absolute value rectification |x|.
- **N**: Local subtractive + divisive normalization (Gaussian neighborhood weights).
- **P_A / P_M**: Average or max pooling + subsampling.

#### 7.2 PSD Unsupervised Filters

Predictive Sparse Decomposition learns fast feed-forward approximations to sparse coding on patches - **encoder** becomes convolutional filter bank.

#### 7.3 Training Protocols

- **R / RR**: random filters, classifier only (or two stages random).
- **U / UU**: unsupervised PSD per stage, fixed filters.
- **R+ / R+R+**: random init then **global supervised** backprop through all stages.
- **U+ / U+U+**: unsupervised init then global supervised refinement.

#### 7.4 Key Empirical Findings

- **Rectification + local normalization** is the dominant accuracy factor on Caltech-101-style setups.
- **Two stages beat one** when paired with sensible pooling/nonlinearities.
- **Random filters** with R_abs-N-PA can reach **~63%** (30 train/class) - **architecture > specific filters** in small-data regime.
- With more labeled data (NORB), **learning** surpasses random - inductive structure cannot fully replace adaptation when data abounds.
- **U+U+** reaches **>65%** on Caltech-101; **NORB** error **5.6%** with best config.
- **MNIST** (undistorted): **0.53%** error with PSD pretraining + supervised refinement - state of art at the time for raw digits.

#### 7.5 Architectural Interpretation

- **Rectification** removes sign ambiguity; avoids cancellation in **average pooling** across opposite-polarity features.
- **Normalization** enforces local competition / variance control - speeds and stabilizes learning.
- **Max pooling** partially substitutes for rectification + average pooling by selecting dominant activations.

> **Modern connection**: This paper is an early systematic case for **ReLU-family** activations and **local response normalization** precursors in deep vision - before AlexNet popularized ReLU at ImageNet scale.

---

## 8) Architectural Synthesis: Core Lessons of the 2000s

### Technical Details

#### 8.1 Comparison Across Architectural Dimensions

| Theme | Representative work | Mechanism | Payoff | Limitation |
| ----- | ------------------- | --------- | ------ | ---------- |
| Word embeddings + neural LM | Bengio 2003 | C matrix + MLP softmax | Smooth generalization, longer context | Softmax compute at huge |V| |
| Undirected pretraining | Hinton 2006 DBN | RBM + CD + stack | Trains deep generative models | Approximate inference; training complexity |
| Deep autoencoder | Hinton & Salakh 2006 | RBM unfold + BP fine-tune | Nonlinear dimensionality reduction | Needs careful protocol for real data |
| Analysis / extensions | Bengio 2007 | Gaussian RBMs, partial supervision | Explains when pretraining helps | Not all tasks benefit equally |
| Robust unsupervised features | Vincent 2008 | Denoise + stack | Strong init without RBMs | Choice of noise distribution |
| Conv generative hierarchy | Lee 2009 | CRBM + prob max-pool | Full-image hierarchical features | Training heavy; mean-field approximations |
| Conv stage engineering | Jarrett 2009 | abs + norm + pool | High accuracy even with random filters | Insights sometimes dataset-dependent |

#### 8.2 Persistent Problems (Agenda for the 2010s)

- **Scaling softmax and attention** in language models -> hierarchical softmax, later Transformers.
- **End-to-end supervised depth** without pretraining -> enabled by **ReLU**, **better init**, **data scale**, **GPUs**.
- **Batch normalization** and **skip connections** -> stabilize very deep discriminative training.
- **Generative evaluation** of deep models -> birth of modern **GANs** and **VAEs** building on representation learning culture.

#### 8.3 Conceptual Bridge to Modern Architecture

```
Neural LM + embedding matrix C    ->  Token embeddings, large-scale LMs
RBM / CD                          ->  Energy models, MCMC approximations (historical)
Greedy layer-wise stacking          ->  Progressive training, curriculum, deep init
Denoising autoencoder               ->  BERT-style mask reconstruction (distant cousin)
Convolutional DBN                 ->  Unsupervised conv feature learning
Rectification + local norm        ->  ReLU, LayerNorm/GroupNorm lineages
```

---

## 9) Critical Points Summary

- **Neural LM (2003)**: Learn **C(i) in R^m** jointly with a neural predictor; generalization comes from **similar words -> similar vectors -> similar predictions**.
- **DBN (2006)**: **RBM stacking** + **CD** gives efficient pretraining; **up-down** adjusts recognition vs generative mismatch.
- **Deep autoencoder (2006)**: **Unfold** pretrained RBMs; global BP fine-tune yields **nonlinear codes** beating PCA.
- **Greedy training study (2007)**: **Unsupervised** pretraining beats **supervised greedy** stacking; explains failure modes of deep nets without init; **partial supervision** fixes "uncooperative" p(x).
- **Denoising autoencoder (2008)**: **Corrupt then reconstruct** forces **manifold-aligned** features; **SdA** competitive with DBNs.
- **CDBN (2009)**: **Convolution + probabilistic max-pooling** scales hierarchical **generative** vision; supports **top-down** completion.
- **Jarrett et al. (2009)**: **abs rectification + local normalization** often matter more than whether filters are random, Gabor, or PSD-learned in small-data regimes; **depth (two stages)** wins.
- **Meta-lesson**: The decade established **representation learning** as a first-class design pattern - pretraining, embeddings, and architectural nonlinearities set the stage for **scale** in the 2010s.
