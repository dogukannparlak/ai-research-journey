# AI Architecture: The 2000s - The Deep Learning Renaissance (Pretraining Era)

These notes cover seven foundational sources:

| Year | Author(s) | Contribution |
|------|-----------|--------------|
| 2003 | Bengio et al. | Neural Probabilistic Language Model (NPLM): learned distributed word representations + neural LM |
| 2006 | Hinton, Osindero, Teh | Deep Belief Nets (DBN): greedy layer-wise pretraining with RBMs, Contrastive Divergence |
| 2006 | Hinton & Salakhutdinov | Deep Autoencoders for dimensionality reduction (pretraining + fine-tuning), nonlinear PCA |
| 2007 | Bengio et al. | Greedy layer-wise training: why unsupervised pretraining helps optimization/generalization |
| 2008 | Vincent et al. | Denoising Autoencoders (DAE): robustness via corruption + reconstruction |
| 2009 | Lee et al. | Convolutional Deep Belief Networks: convolution + probabilistic pooling in a generative model |
| 2009 | Jarrett et al. | Multi-stage object recognition: architectural + activation choices (ReLU/rectification), pooling/normalization |

> Note on local sources: the PDFs currently stored in `History/2000s/` appear to be corrupted placeholders (each file is ~131 bytes). The content below reflects the standard, canonical technical contributions of the listed papers rather than verbatim excerpts from the local PDFs.

---

## 1) Historical Context: Why the 2000s Brought Deep Learning Back

By the late 1990s, backpropagation, CNNs (LeNet), and LSTMs existed, but **training deep models reliably** was still hard and compute/data were limited. In practice, many problems were dominated by:

- **Kernel methods (SVMs)** and carefully engineered features
- **Shallow** models that were easier to optimize
- A persistent optimization barrier: **vanishing gradients** + poor local minima behavior in deep networks

The 2000s shift is not just “more layers”—it is the emergence of a pragmatic recipe:

1. **Unsupervised (or self-supervised) layer-wise pretraining** to initialize deep networks in a good region of parameter space
2. **Supervised fine-tuning** with backpropagation end-to-end
3. New representation-learning ideas that later become standard: **embeddings**, **autoencoders**, **denoising**, and **convolutional feature hierarchies**

---

## 2) Neural Probabilistic Language Model (2003) — Bengio et al.

### Core Architectural Idea
Language modeling can be approached as:

> Learn a continuous representation of words (embeddings) and use a neural network to model \(P(w_t \mid w_{t-n+1:t-1})\).

This solves a key sparsity problem in NLP: **one-hot** representations do not express similarity. Embeddings make similar words close in a learned vector space, allowing generalization to unseen \(n\)-grams.

### Technical Details

#### 2.1 Parameterization with Word Embeddings
Let the vocabulary size be \(V\), embedding dimension be \(d\).

- Each word \(w\) has an embedding \(e(w) \in \mathbb{R}^d\)
- For context length \(n-1\), concatenate:

```
x = [e(w_{t-n+1}); e(w_{t-n+2}); ...; e(w_{t-1})]  ∈ R^{(n-1)d}
```

Then use a feedforward network:

```
h = tanh(W_h x + b_h)
z = W_o h + b_o
P(w_t = i | context) = softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

Train by maximizing log-likelihood (cross-entropy) over a corpus:

```
L = - Σ_t log P(w_t | w_{t-n+1:t-1})
```

#### 2.2 Key Architectural Consequence
- **Representation learning** for discrete symbols: the embedding matrix is learned jointly with the predictor.
- The model defines a smooth function over contexts, enabling **distributed generalization**.

> Modern connection: This is a direct ancestor of Word2Vec/GloVe (embeddings as a primary object) and of modern neural LMs/Transformers (softmax over vocabulary + learned token vectors).

---

## 3) Deep Belief Nets (2006) — Hinton, Osindero, Teh

### Core Architectural Idea
Deep networks can be trained effectively by **greedy layer-wise pretraining** using generative models.

The DBN recipe:

1. Train first layer as an **RBM** on data
2. Freeze it, transform data into hidden activations
3. Train the next RBM on those activations
4. Stack multiple layers
5. Optionally fine-tune with supervised backprop

### Technical Details

#### 3.1 Restricted Boltzmann Machine (RBM)
Binary visible units \(v\) and hidden units \(h\) with energy:

```
E(v,h) = -b^T v - c^T h - v^T W h
```

Joint distribution:

```
P(v,h) = (1/Z) exp(-E(v,h))
```

Because the graph is bipartite (no v–v or h–h edges), conditionals factorize:

```
P(h_j=1|v) = sigmoid(c_j + Σ_i W_{ij} v_i)
P(v_i=1|h) = sigmoid(b_i + Σ_j W_{ij} h_j)
```

#### 3.2 Contrastive Divergence (CD-k)
Exact likelihood gradients require intractable model expectations. CD approximates:

```
ΔW ∝ <v h^T>_data  -  <v h^T>_model
```

with a short Gibbs chain starting at the data:

```
v0 (data) → h0 ~ P(h|v0) → v1 ~ P(v|h0) → h1 ~ P(h|v1) → ...
```

CD-1 uses only one step. This makes pretraining practical.

#### 3.3 Why This Was Architecturally Important
- Pretraining provides **better initialization** than random weights for deep nets.
- Each layer learns features that explain the layer below, forming a hierarchy.

> Modern connection: Layer-wise pretraining is a predecessor of modern self-supervised pretraining (though the objectives differ). The big idea is the same: learn representations from unlabeled data first, then fine-tune.

---

## 4) Reducing Dimensionality with Neural Networks (2006) — Hinton & Salakhutdinov

### Core Architectural Idea
Use a **deep autoencoder** to compress data into a low-dimensional code, achieving nonlinear dimensionality reduction that can outperform linear PCA on complex manifolds.

Key recipe:

1. Pretrain a deep autoencoder by stacking RBMs (as in DBNs)
2. “Unroll” into an encoder + decoder
3. Fine-tune with backprop to minimize reconstruction error

### Technical Details

#### 4.1 Autoencoder Objective
Encoder \(f_\theta\) maps input to code, decoder \(g_\phi\) reconstructs:

```
z = f_θ(x)
ŷ = g_φ(z)
```

Reconstruction loss (example: MSE for real-valued data):

```
L = ||x - ŷ||^2
```

For binary data, a cross-entropy reconstruction loss is common.

#### 4.2 Nonlinear “PCA-like” Compression
If the code dimension \(k\) is small, the network must learn a compact representation:

- PCA: linear subspace
- Deep autoencoder: nonlinear manifold (piecewise nonlinear due to activations)

> Modern connection: This line of work is a direct ancestor of representation learning with autoencoders and later VAEs; it also foreshadows today’s use of large encoders as general-purpose compressors of information.

---

## 5) Greedy Layer-wise Training of Deep Networks (2007) — Bengio et al.

### Core Architectural Idea
Unsupervised pretraining helps deep learning for two reasons:

1. **Optimization**: it sets parameters into a region where gradients carry signal and bad local minima are avoided
2. **Regularization**: it biases the model toward representations that capture structure in \(P(x)\), improving generalization

This is a conceptual bridge between the 1990s (vanishing gradients, shallow successes) and the 2010s (end-to-end deep learning).

### Technical Details

#### 5.1 Greedy Layer-wise Procedure (Generic Form)
For layers \(1..L\):

```
Train layer l to model its input (unsupervised objective)
Freeze layer l
Compute representation for next layer
```

After stacking:

```
Initialize supervised network with pretrained weights
Fine-tune all layers with backprop on labels
```

#### 5.2 Key Takeaway
- Pretraining is not merely “extra training”; it changes the geometry of the optimization problem by placing the network in a better basin.

> Modern connection: Pretraining becomes the dominant paradigm again with Transformers—now via self-supervised objectives (masked LM, next-token prediction), but for the same deep reason: learn good internal representations before task-specific training.

---

## 6) Denoising Autoencoders (2008) — Vincent et al.

### Core Architectural Idea
Instead of reconstructing the input from itself (which can learn identity), force robustness:

> Corrupt the input \(x\) into \(\tilde{x}\), then train the model to reconstruct the original \(x\).

This encourages the representation to capture stable structure, not noise.

### Technical Details

#### 6.1 Corruption + Reconstruction Objective
Let \(q(\tilde{x}|x)\) be a corruption process (masking noise, Gaussian noise, salt-and-pepper).

```
tilde_x ~ q(tilde_x | x)
z = f_θ(tilde_x)
ŷ = g_φ(z)
Minimize:  L = E_{tilde_x~q}[ loss(x, ŷ) ]
```

#### 6.2 Why This Matters Architecturally
- The encoder learns features that are stable under perturbations.
- DAEs can be stacked (like pretraining) to build deep representations.

> Modern connection: Denoising objectives are core to diffusion models (denoise to recover signal) and to robustness-aware representation learning.

---

## 7) Convolutional Deep Belief Networks (2009) — Lee et al.

### Core Architectural Idea
Bring together:

- **Convolution** (local receptive fields + weight sharing)
- **Generative pretraining** (RBM/DBN-style)
- **Pooling** (spatial invariance)

but in a probabilistic/generative framework rather than purely discriminative CNNs.

### Technical Details

#### 7.1 Convolutional RBM Intuition
Instead of a dense matrix \(W\), use filters \(W_k\) convolved across the image:

```
FeatureMap_k = conv2d(X, W_k) + b_k
```

Hidden activations correspond to feature detections at positions.

#### 7.2 Probabilistic Pooling (High Level)
Pooling provides invariance by aggregating activations within a local region. In probabilistic pooling, the model can represent a latent “pooled” variable that explains which location in a pooling block is active, rather than deterministically taking max/avg.

Architectural consequence:
- Build deeper convolutional hierarchies with generative pretraining
- Learn features without labels, then fine-tune for recognition

> Modern connection: While modern CNNs are typically trained end-to-end discriminatively, the idea of combining convolution + unsupervised learning reappears in self-supervised vision.

---

## 8) Multi-stage Architecture for Object Recognition (2009) — Jarrett et al.

### Core Architectural Idea
Many “small” architectural choices dominate performance:

- Nonlinearities (rectification vs. tanh)
- Pooling choice and placement
- Local contrast normalization / divisive normalization
- Depth and stage composition

### Technical Details

#### 8.1 Rectification (ReLU-like) vs. Saturating Nonlinearities
Saturating nonlinearities (sigmoid/tanh) can reduce gradient flow. Rectification:

```
ReLU(x) = max(0, x)
```

preserves gradient for positive activations and encourages sparse activations.

#### 8.2 Typical Multi-stage Pipeline (Conceptual)

```
Input
 -> [Conv -> Rectification -> (Local normalization) -> Pool]
 -> [Conv -> Rectification -> (Local normalization) -> Pool]
 -> Classifier
```

Key outcome: performance sensitivity shows that “architecture engineering” is crucial even before ImageNet-era scaling.

> Modern connection: This paper anticipates the dominance of ReLU/rectifiers and the importance of normalization + pooling decisions in modern convnets.

---

## 9) Architectural Synthesis: What the 2000s Added

### Technical Details

#### 9.1 Comparison Across Architectural Dimensions

| Architecture | Main mechanism | Learned representation | Key advantage | Main limitation (historical) |
|-------------|----------------|------------------------|--------------|------------------------------|
| NPLM (2003) | embeddings + FFNN + softmax | continuous word vectors | generalization across discrete tokens | expensive softmax / limited context length |
| DBN/RBM (2006) | greedy generative pretraining | hierarchical latent features | makes deep training practical | generative training complexity; later replaced by simpler pretraining |
| Deep AE (2006) | pretraining + fine-tuning reconstruction | nonlinear low-dim code | nonlinear dimensionality reduction | reconstruction not always aligned with task |
| Greedy pretraining theory (2007) | optimization + regularization lens | improved basins | explains why pretraining helps | later less needed with ReLU/BN/scale |
| DAE (2008) | corruption + reconstruction | robust features | discourages identity mapping | objective choice matters |
| Conv DBN (2009) | conv + probabilistic pooling | invariant feature hierarchy | unlabeled feature learning for vision | computational complexity |
| Multi-stage recognition (2009) | rectification/normalization/pooling | better discriminative features | shows “details matter” | still pre-ImageNet scale |

#### 9.2 Persistent Problems (Agenda of the Next Era)
The 2000s solution relied heavily on pretraining because deep supervised training still faced:

- Optimization fragility (especially with saturating activations)
- Limited labeled data at scale
- Lack of GPU-centric training pipelines

The 2010s will flip the default to end-to-end supervised/self-supervised training at scale (ImageNet + GPUs), but many 2000s concepts remain central.

#### 9.3 Bridge to Modern Architecture

```
NPLM embeddings                 -> token embeddings in Transformers
Greedy unsupervised pretraining -> foundation-model pretraining (self-supervised)
Deep autoencoders               -> VAEs, representation learning, compression
Denoising objectives            -> diffusion models, robustness, masked modeling
Conv feature hierarchies        -> modern CNNs and vision backbones
Rectifiers + normalization      -> ReLU-family + normalization as defaults
```

---

## 10) Critical Points Summary

- **Embeddings (2003)**: discrete symbols become continuous vectors learned jointly with prediction; generalization improves dramatically.
- **DBN/RBM pretraining (2006)**: greedy layer-wise unsupervised training breaks the “deep nets don’t train” barrier of the time.
- **Deep autoencoders (2006)**: nonlinear dimensionality reduction becomes practical with pretraining + fine-tuning.
- **Why pretraining helps (2007)**: it improves optimization and acts as a regularizer by capturing structure in \(P(x)\).
- **Denoising (2008)**: corruption + reconstruction encourages robust, meaningful features rather than identity.
- **Conv + generative modeling (2009)**: convolutional hierarchies can be learned with limited labels via unsupervised objectives.
- **Rectification matters (2009)**: activation/normalization/pooling choices can dominate performance—foreshadowing ReLU-era deep learning.

