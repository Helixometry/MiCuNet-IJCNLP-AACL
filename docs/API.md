# MiCuNet API Reference

## Core Model

### `MiCuNet`

Main model class for emotional manipulation detection in synthetic speech.

```python
from micunet.model import MiCuNet

model = MiCuNet(
    input_dim=768,
    hidden_dim=512,
    num_emotions=7,
    num_sources=3,
    dropout=0.3
)
```

#### Parameters
- `input_dim` (int): Dimension of input features (default: 768)
- `hidden_dim` (int): Hidden layer dimension (default: 512)
- `num_emotions` (int): Number of emotion classes (default: 7)
- `num_sources` (int): Number of source/generator classes (default: 3)
- `dropout` (float): Dropout probability (default: 0.3)

#### Methods

##### `forward(x: torch.Tensor) -> Dict[str, torch.Tensor]`
Forward pass through the model.

```python
import torch
x = torch.randn(4, 768)  # Batch of 4 samples
outputs = model(x)
# outputs contains:
# - 'original_emotion': (4, 7)
# - 'manipulated_emotion': (4, 7)
# - 'source': (4, 3)
# - 'features': (4, 512)
```

##### `from_pretrained(checkpoint_path: str) -> MiCuNet`
Load a pretrained model from checkpoint.

```python
model = MiCuNet.from_pretrained('models/checkpoint.pt')
```

##### `save(checkpoint_path: str, config: Optional[Dict] = None)`
Save model to checkpoint.

```python
model.save('models/checkpoint.pt', config={'input_dim': 768})
```

## Layers

### `MixedCurvatureProjection`

Mixed-curvature projection layer spanning Hyperbolic, Euclidean, and Spherical spaces.

```python
from micunet.layers import MixedCurvatureProjection

layer = MixedCurvatureProjection(input_dim=768, output_dim=512)
output = layer(input_tensor)  # (batch_size, 512)
```

#### Parameters
- `input_dim` (int): Input feature dimension
- `output_dim` (int): Output feature dimension
- `curvature_weight` (float): Weight for curvature balancing

### `TemporalGating`

Learnable temporal gating mechanism for adaptive feature weighting.

```python
from micunet.layers import TemporalGating

layer = TemporalGating(feature_dim=512, gating_dim=128)
output = layer(input_tensor)  # Same shape as input
```

### `MultiHeadTemporalAttention`

Multi-head attention for temporal feature aggregation.

```python
from micunet.layers import MultiHeadTemporalAttention

layer = MultiHeadTemporalAttention(feature_dim=512, num_heads=8)
output = layer(query, key, value)
```

## Encoders

### `DualStreamEncoder`

Combines speech foundation model and spectrogram encoders.

```python
from micunet.encoders import DualStreamEncoder

encoder = DualStreamEncoder(
    foundation_model_dim=1024,
    n_mels=128,
    output_dim=512,
    fusion_type='concat'
)

fm_embeddings = torch.randn(4, 1024)
spectrogram = torch.randn(4, 128, 100)
output = encoder(fm_embeddings, spectrogram)  # (4, 512)
```

### `SpeechFoundationModelEncoder`

Encodes speech foundation model embeddings.

```python
from micunet.encoders import SpeechFoundationModelEncoder

encoder = SpeechFoundationModelEncoder(input_dim=1024, output_dim=512)
output = encoder(embeddings)
```

### `SpectrogramEncoder`

Processes spectrogram features using convolutions.

```python
from micunet.encoders import SpectrogramEncoder

encoder = SpectrogramEncoder(n_mels=128, output_dim=512)
output = encoder(spectrogram)  # (batch_size, 512)
```

## Data Loading

### `AudioDataLoader`

Loads audio files with optional labels.

```python
from data.loaders import AudioDataLoader

loader = AudioDataLoader(
    audio_dir='data/english/train',
    labels_file='data/english/train/labels.csv',
    batch_size=32,
    shuffle=True
)

for batch in loader:
    audio = batch['audio']
    labels = batch['original_emotion']
```

### `AudioDataset`

PyTorch Dataset for audio files.

```python
from data.loaders import AudioDataset
import torch.utils.data

dataset = AudioDataset(
    audio_dir='data/english/train',
    labels_file='labels.csv',
    sr=16000,
    duration=3.0
)

loader = torch.utils.data.DataLoader(dataset, batch_size=32)
```

## Audio Processing

### `AudioPreprocessor`

Audio preprocessing pipeline.

```python
from data.preprocessor import AudioPreprocessor

preprocessor = AudioPreprocessor(sr=16000, n_mels=128)

# Load audio
audio = preprocessor.load_audio('path/to/audio.wav')

# Extract features
mfcc = preprocessor.extract_mfcc(audio)
melspec = preprocessor.extract_melspectrogram(audio)
chroma = preprocessor.extract_chroma(audio)
zcr = preprocessor.extract_zero_crossing_rate(audio)

# Normalize
normalized, mean, std = preprocessor.normalize(features)

# Full pipeline
features_dict = preprocessor.process('path/to/audio.wav')
```

## Utility Functions

### `micunet.utils`

```python
from micunet.utils import (
    load_audio,
    extract_spectrogram,
    normalize_features,
    get_device,
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
)

# Load audio
audio = load_audio('path/to/audio.wav', sr=16000)

# Extract spectrogram
spec = extract_spectrogram(audio, n_mels=128)

# Get device
device = get_device()  # Returns 'cuda' or 'cpu'

# Set seeds for reproducibility
seed_everything(seed=42)

# Save and load checkpoints
save_checkpoint(model, optimizer, epoch, loss, 'checkpoint.pt')
checkpoint = load_checkpoint('checkpoint.pt', model, optimizer)

# Track metrics
meter = AverageMeter()
meter.update(loss_value, batch_size)
print(meter.avg)  # Average value
```

## Configuration

Configurations are YAML files in `configs/` directory:

```yaml
model:
  input_dim: 768
  hidden_dim: 512
  num_emotions: 7
  num_sources: 3

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  weight_decay: 1e-5
```

## Training Script

```bash
python scripts/train.py --config configs/config_english.yaml
```

## Inference Script

```bash
python scripts/inference.py \
    --audio path/to/audio.wav \
    --checkpoint models/checkpoint.pt \
    --language en
```

## Evaluation Script

```bash
python scripts/evaluate.py \
    --checkpoint models/checkpoint.pt \
    --test-dir data/english/test \
    --output-dir results/evaluation
```

## Example Usage

```python
import torch
from micunet.model import MiCuNet
from data.preprocessor import AudioPreprocessor

# Load model
model = MiCuNet.from_pretrained('models/checkpoint.pt')
model.eval()

# Prepare audio
preprocessor = AudioPreprocessor()
audio = preprocessor.load_audio('path/to/audio.wav')
melspec = preprocessor.extract_melspectrogram(audio)

# Run inference
with torch.no_grad():
    inputs = torch.from_numpy(melspec).float().unsqueeze(0)
    outputs = model(inputs)

# Get predictions
original_emotion = outputs['original_emotion'][0].argmax().item()
manipulated_emotion = outputs['manipulated_emotion'][0].argmax().item()
source = outputs['source'][0].argmax().item()

print(f"Original: {original_emotion}, Manipulated: {manipulated_emotion}, Source: {source}")
```

## Error Handling

```python
try:
    model = MiCuNet.from_pretrained('non_existent.pt')
except FileNotFoundError:
    print("Checkpoint not found")

try:
    output = model(torch.randn(4, 512))  # Wrong input dim
except RuntimeError as e:
    print(f"Runtime error: {e}")
```
