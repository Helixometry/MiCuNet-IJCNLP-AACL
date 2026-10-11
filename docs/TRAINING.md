# Training Guide for MiCuNet

## Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU training, optional)

## Installation

```bash
# Clone repository
git clone https://github.com/gir-ish/MiCuNet-IJCNLP-AACL.git
cd MiCuNet-IJCNLP-AACL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### Emo-Fake Dataset

1. Download the Emo-Fake dataset from the official source
2. Organize directory structure:
   ```
   data/
   ├── english/
   │   ├── train/
   │   │   ├── audio_files/
   │   │   └── labels.csv
   │   ├── val/
   │   └── test/
   └── chinese/
       ├── train/
       ├── val/
       └── test/
   ```

3. Labels CSV format:
   ```csv
   audio_path,original_emotion,manipulated_emotion,source
   audio1.wav,0,1,0
   audio2.wav,2,3,1
   ...
   ```

   Emotion classes:
   - 0: Neutral
   - 1: Happiness
   - 2: Sadness
   - 3: Anger
   - 4: Surprise
   - 5: Fear
   - 6: Disgust

## Configuration

Edit `configs/config_english.yaml` or `configs/config_chinese.yaml`:

```yaml
# Model architecture
model:
  input_dim: 768
  hidden_dim: 512
  num_emotions: 7
  num_sources: 3
  dropout: 0.3

# Training parameters
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  weight_decay: 1e-5
  
# Data
data:
  train_dir: data/english/train
  val_dir: data/english/val
  test_dir: data/english/test
  sr: 16000
  duration: 3.0

# Device
device: cuda  # or cpu
seed: 42
```

## Training

### Basic Training

```bash
python scripts/train.py --config configs/config_english.yaml
```

### With Custom Parameters

```bash
python scripts/train.py \
  --config configs/config_english.yaml \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.0005 \
  --output-dir results/my_experiment
```

### Resume Training

```bash
python scripts/train.py \
  --config configs/config_english.yaml \
  --resume models/checkpoint_epoch_30.pt
```

## Monitoring Training

### TensorBoard

```bash
# During training
tensorboard --logdir=results/logs

# Open browser at http://localhost:6006
```

### Metrics Tracked

- Training loss (overall, per task)
- Validation loss
- F1 scores (original emotion, manipulated emotion, source)
- Learning rate schedule

## Evaluation

### Evaluate on Test Set

```bash
python scripts/evaluate.py \
  --checkpoint models/english_checkpoint.pt \
  --test-dir data/english/test \
  --output-dir results/evaluation
```

### Metrics

The evaluation script reports:
- **Precision, Recall, F1-score** (macro and weighted)
- **Confusion matrices** for each task
- **Per-class metrics**

## Cross-Lingual Training

### Training on English

```bash
python scripts/train.py --config configs/config_english.yaml
```

### Fine-tuning on Chinese

```bash
python scripts/train.py \
  --config configs/config_chinese.yaml \
  --pretrained models/english_checkpoint.pt \
  --freeze-encoder \
  --epochs 30 \
  --learning-rate 0.0001
```

## Inference

### Single Audio File

```bash
python scripts/inference.py \
  --audio path/to/audio.wav \
  --checkpoint models/english_checkpoint.pt \
  --language en
```

### Batch Processing

```python
from micunet.model import MiCuNet
from data.preprocessor import AudioPreprocessor
import torch

# Load model
model = MiCuNet.from_pretrained('models/english_checkpoint.pt')
model.eval()

# Load audio
preprocessor = AudioPreprocessor()
audio = preprocessor.load_audio('path/to/audio.wav')

# Extract features
melspec = preprocessor.extract_melspectrogram(audio)

# Inference
with torch.no_grad():
    output = model(torch.from_numpy(melspec).float().unsqueeze(0))

print("Original Emotion:", output['original_emotion'].argmax())
print("Manipulated Emotion:", output['manipulated_emotion'].argmax())
print("Source:", output['source'].argmax())
```

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| learning_rate | 0.001 | [1e-4, 1e-2] | Start with default |
| batch_size | 32 | [16, 64] | Larger = more stable gradients |
| dropout | 0.3 | [0.1, 0.5] | Prevent overfitting |
| hidden_dim | 512 | [256, 1024] | Model capacity |
| weight_decay | 1e-5 | [1e-6, 1e-4] | L2 regularization |

### Recommended Grid Search

```bash
# Script for grid search
python scripts/grid_search.py \
  --config configs/config_english.yaml \
  --learning-rates 0.0005 0.001 0.002 \
  --batch-sizes 16 32 64 \
  --dropouts 0.1 0.3 0.5
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size in config
batch_size: 16  # or lower

# Or use gradient accumulation
gradient_accumulation_steps: 2
```

### Poor Performance

1. Check data preprocessing
2. Verify class balance
3. Increase training data
4. Adjust learning rate schedule
5. Try different random seed

### Training Not Converging

```yaml
# Increase model capacity
hidden_dim: 1024

# Reduce dropout
dropout: 0.1

# Adjust learning rate
learning_rate: 0.0001
```

## Best Practices

1. **Always use a validation set** to monitor for overfitting
2. **Log metrics regularly** using TensorBoard
3. **Save checkpoints** at regular intervals
4. **Use same preprocessing** for train/val/test
5. **Normalize input features** before feeding to model
6. **Use stratified splits** for imbalanced classes
7. **Set random seed** for reproducibility

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{girish2025micunet,
  title={Towards Attribution of Generators and Emotional Manipulation in Cross-Lingual Synthetic Speech using Geometric Learning},
  author={Girish and Akhtar, Mohd Mujtaba and Sheth, Farhan and Singh, Muskaan},
  booktitle={Findings of the IJCNLP-AACL 2025},
  year={2025}
}
```
