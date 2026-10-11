# MiCuNet Architecture

## Overview

MiCuNet (Multi-Curvature Learning Network) is a novel deep learning architecture designed for detecting emotional manipulation in synthetic speech. The key innovation is the use of **mixed-curvature geometry** spanning Hyperbolic, Euclidean, and Spherical spaces.

## Core Components

### 1. Dual-Stream Feature Extraction

The model processes two complementary feature streams:

#### Stream 1: Speech Foundation Model Embeddings
- Extracts semantic and prosodic information from pre-trained speech models
- Captures high-level linguistic and emotional content
- Dimension: 1024 → 512 (after encoding)

#### Stream 2: Spectrogram Features
- Processes raw spectral information using convolutional layers
- Captures low-level acoustic patterns
- 3 convolutional blocks with batch normalization and pooling
- Output dimension: 512

### 2. Mixed-Curvature Projection Mechanism

The core innovation that enables capture of complex geometric relationships:

```
Input Features (768)
        ↓
    ┌───┴───┬────────────┐
    ↓       ↓            ↓
Euclidean Hyperbolic Spherical
  Space    Space      Space
    ↓       ↓            ↓
  Linear  Tanh        Linear
         (Non-linear) (Normalized)
    ↓       ↓            ↓
    └───┬───┴────────────┘
        ↓
  Weighted Sum
  (Learnable α, β, γ)
        ↓
  Mixed-Curvature
   Features (512)
```

#### Euclidean Projection
- Standard linear transformation: `W_e * x + b_e`
- Captures standard feature relationships

#### Hyperbolic Projection
- Non-linear transformation: `tanh(W_h * x + b_h)`
- Models hierarchical structures in emotion classification
- Better represents tree-like relationships

#### Spherical Projection
- Normalized linear transformation: `normalize(W_s * x + b_s)`
- Captures directional relationships
- Useful for angle-based emotion similarity

### 3. Temporal Gating Mechanism

Applies learnable attention-based gating for adaptive feature weighting:

```
Input (batch_size, seq_len, feature_dim)
        ↓
  Dense(feature_dim → gating_dim)
        ↓
      ReLU
        ↓
  Dense(gating_dim → feature_dim)
        ↓
     Sigmoid
        ↓
  Element-wise Multiplication
        ↓
   Gated Output
```

Benefits:
- Focuses on relevant time steps
- Suppresses irrelevant temporal regions
- Adaptive to different audio lengths

### 4. Multitask Learning Heads

Three independent classification heads for simultaneous prediction:

#### Original Emotion Head
- Classifies the original emotion in source audio
- Classes: Happiness, Sadness, Anger, Neutral, Surprise, Fear, Disgust (7 classes)

#### Manipulated Emotion Head
- Classifies the emotion after manipulation
- Same output space as original emotion

#### Source Identification Head
- Identifies the manipulation source/generator
- Classes: TTS Model A, TTS Model B, TTS Model C (3 classes)

## Data Flow

```
Audio Input
    ↓
┌─────────────────────────────────────┐
│   Feature Extraction                │
├─────────────────────────────────────┤
│  Foundation Model   │  Spectrogram  │
│  Embeddings         │  Features     │
│  (1024)             │  (128 mels)   │
└──────┬──────────────┬────────────────┘
       ↓              ↓
┌──────────────────────────────────┐
│  Dual-Stream Encoder             │
│  FM Encoder    │  Spec Encoder   │
│  (512)         │  (512)          │
└──────┬─────────┬─────────────────┘
       ↓         ↓
┌──────────────────────────────────┐
│  Feature Fusion (concat/add)      │
│  Output: 512 or 1024             │
└──────┬──────────────────────────┘
       ↓
┌──────────────────────────────────┐
│  Mixed-Curvature Projection      │
│  (Hyp + Eucl + Spher)            │
│  Output: 512                      │
└──────┬──────────────────────────┘
       ↓
┌──────────────────────────────────┐
│  Temporal Gating                  │
│  Output: 512                      │
└──────┬──────────────────────────┘
       ↓
   ┌───┴────┬───────────┐
   ↓        ↓           ↓
Original Manipulated Source
Emotion  Emotion      ID
(7-way)  (7-way)    (3-way)
```

## Mathematical Formulation

### Mixed-Curvature Projection

For input features `x`:

```
y_e = W_e @ x + b_e              (Euclidean)
y_h = tanh(W_h @ x + b_h)        (Hyperbolic)
y_s = normalize(W_s @ x + b_s)   (Spherical)

output = α * y_e + β * y_h + γ * y_s

where α + β + γ ≤ 1 (normalized)
```

### Temporal Gating

```
gate = σ(W_2 * relu(W_1 * x + b_1) + b_2)
output = x ⊙ gate

where σ is sigmoid, ⊙ is element-wise multiplication
```

## Model Parameters

- Total parameters: ~2.3M
- Trainable parameters: ~2.3M

### Configuration

```yaml
input_dim: 768              # Combined feature dimension
hidden_dim: 512             # Hidden layer dimension
num_emotions: 7             # Emotion classes
num_sources: 3              # Source/generator classes
dropout: 0.3                # Dropout rate
learning_rate: 1e-3         # Adam optimizer
batch_size: 32
epochs: 50
```

## Training Strategy

### Loss Function
- Multi-task loss combining three classification tasks:
  ```
  Loss = λ₁ * L_original + λ₂ * L_manipulated + λ₃ * L_source
  ```
  where λ₁ = λ₂ = λ₃ = 1/3 (equal weighting)

### Optimization
- Optimizer: Adam (lr=0.001, β₁=0.9, β₂=0.999)
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Batch size: 32
- Epochs: 50

### Regularization
- Dropout: 0.3
- Batch normalization
- L2 regularization (weight decay=1e-5)

## Performance Characteristics

### Computational Complexity
- Inference time per sample: ~50ms (GPU) / ~200ms (CPU)
- Memory requirements: ~2GB (GPU) / ~1GB (CPU)

### Cross-lingual Transfer
- Pre-trained on English, fine-tuned on Chinese
- Performance degradation: ~2% F1 score

## Future Improvements

1. **Curvature-Adaptive Learning**: Make curvatures learnable based on input
2. **Attention Mechanisms**: Add multi-head attention for feature fusion
3. **Domain Adaptation**: Techniques for unseen TTS models
4. **Knowledge Distillation**: Compress model for deployment

## References

- Hyperbolic Neural Networks: https://arxiv.org/abs/1805.09112
- Speech Foundation Models: https://huggingface.co/spaces/models
- Multitask Learning: https://arxiv.org/abs/1708.02412
