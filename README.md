# MiCuNet: Multi-Curvature Learning for Emotional Manipulation Detection in Synthetic Speech

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ACL%202025-blue)](https://aclanthology.org/2025.findings-ijcnlp.37/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

A curvature-adaptive deep learning framework for detecting emotional manipulation in cross-lingual synthetic speech

</div>

## Overview

**MiCuNet** is a novel multitask deep learning framework designed to simultaneously identify three critical aspects of synthetically manipulated speech:
- **Original emotions** in the source audio
- **Manipulated emotions** introduced through synthesis
- **Source identification** of the emotional manipulation

The key innovation is a **mixed-curvature projection mechanism** that seamlessly spans Hyperbolic, Euclidean, and Spherical spaces, guided by a learnable temporal gating mechanism. This architecture enables the model to capture the complex geometric relationships inherent in emotional and spectral features.

### Key Features

✨ **Curvature-Adaptive Architecture**: First framework to leverage mixed-curvature geometry (Hyperbolic, Euclidean, Spherical) for synthetic speech detection

🌍 **Cross-Lingual Support**: Evaluated on English and Chinese languages from the Emo-Fake dataset

📊 **Multitask Learning**: Simultaneously detects original emotions, manipulated emotions, and source identification

🔊 **Foundation Model Integration**: Leverages Speech Foundation Models (semantic-prosodic embeddings) combined with spectrogram features

⚡ **Temporal Gating**: Learnable temporal gating mechanism for adaptive feature weighting

## Paper

**Towards Attribution of Generators and Emotional Manipulation in Cross-Lingual Synthetic Speech using Geometric Learning**

- **Authors**: Girish, Mohd Mujtaba Akhtar, Farhan Sheth, Muskaan Singh
- **Conference**: Findings of IJCNLP-AACL 2025
- **Paper Link**: [ACL Anthology](https://aclanthology.org/2025.findings-ijcnlp.37/)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gir-ish/MiCuNet-IJCNLP-AACL.git
cd MiCuNet-IJCNLP-AACL

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from micunet.model import MiCuNet
from micunet.utils import load_audio

# Load the pretrained model
model = MiCuNet.from_pretrained('path/to/checkpoint')

# Process audio
audio_path = "path/to/audio.wav"
audio = load_audio(audio_path)

# Get predictions
predictions = model(audio)
print(f"Original Emotion: {predictions['original_emotion']}")
print(f"Manipulated Emotion: {predictions['manipulated_emotion']}")
print(f"Source: {predictions['source']}")
```

## Repository Structure

```
MiCuNet-IJCNLP-AACL/
├── micunet/                    # Main package
│   ├── __init__.py
│   ├── model.py               # Core MiCuNet model
│   ├── layers.py              # Mixed-curvature projection layers
│   ├── encoders.py            # Feature encoders
│   └── utils.py               # Utility functions
├── data/                       # Dataset scripts and utilities
│   ├── __init__.py
│   ├── loaders.py             # Data loading utilities
│   ├── preprocessor.py        # Audio preprocessing
│   └── emo_fake/              # Emo-Fake dataset scripts
├── models/                     # Pretrained model checkpoints
│   ├── english_checkpoint.pt
│   └── chinese_checkpoint.pt
├── configs/                    # Configuration files
│   ├── config_english.yaml
│   ├── config_chinese.yaml
│   └── default_config.yaml
├── scripts/                    # Training and evaluation scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── inference.py          # Inference script
│   └── download_data.py      # Dataset download utilities
├── results/                    # Results and logs
│   ├── metrics/               # Performance metrics
│   └── logs/                  # Training logs
├── tests/                      # Unit tests
│   ├── test_model.py
│   ├── test_layers.py
│   └── test_data.py
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md        # Detailed architecture
│   ├── TRAINING.md            # Training guide
│   └── API.md                 # API reference
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## Model Architecture

### Mixed-Curvature Projection Mechanism

The core innovation of MiCuNet is its mixed-curvature architecture:

1. **Feature Extraction**: 
   - Speech Foundation Model embeddings (semantic-prosodic)
   - Spectrogram features

2. **Curvature-Adaptive Projection**:
   - **Hyperbolic Space**: Captures hierarchical emotion structures
   - **Euclidean Space**: Standard linear transformations
   - **Spherical Space**: Models directional relationships

3. **Temporal Gating**: 
   - Learnable attention mechanism
   - Adaptive weighting of temporal features

4. **Multitask Head**:
   - Original emotion classification
   - Manipulated emotion classification
   - Source identification

For detailed architecture information, see [ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Results

### Performance on Emo-Fake Dataset

| Language | Original Emotion | Manipulated Emotion | Source ID | Avg F1 |
|----------|-----------------|-------------------|-----------|--------|
| English  | 85.4%           | 82.1%             | 88.3%     | 85.3%  |
| Chinese  | 83.2%           | 79.8%             | 86.1%     | 83.0%  |

*Results show F1-scores for multitask learning evaluation*

## Getting Started

### Training

```bash
python scripts/train.py --config configs/config_english.yaml
```

For detailed training guide, see [TRAINING.md](docs/TRAINING.md)

### Evaluation

```bash
python scripts/evaluate.py --checkpoint models/english_checkpoint.pt --test-data path/to/test/data
```

### Inference

```bash
python scripts/inference.py --audio path/to/audio.wav --checkpoint models/english_checkpoint.pt
```

## Dataset

The model is evaluated on the **Emo-Fake** dataset:
- Cross-lingual synthetic speech samples (English & Chinese)
- Manipulated emotional content
- Multiple emotion categories
- Source speaker annotations

To download the dataset:
```bash
python scripts/download_data.py --dataset emo-fake --language en zh
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchaudio 0.9+
- librosa 0.9+
- NumPy, SciPy
- PyYAML (for config management)

See `requirements.txt` for complete dependencies.

## Installation from Source

```bash
git clone https://github.com/gir-ish/MiCuNet-IJCNLP-AACL.git
cd MiCuNet-IJCNLP-AACL
pip install -e .
```

## Usage Examples

### Example 1: Batch Processing

```python
from micunet.model import MiCuNet
from data.loaders import AudioDataLoader

model = MiCuNet.from_pretrained('models/english_checkpoint.pt')
loader = AudioDataLoader('path/to/audio/directory')

for batch in loader:
    predictions = model(batch)
    # Process predictions
```

### Example 2: Custom Configuration

```python
from micunet.model import MiCuNet
import yaml

with open('configs/custom_config.yaml') as f:
    config = yaml.safe_load(f)

model = MiCuNet(config)
# Train or use the model
```

## Citation

If you use MiCuNet in your research, please cite the following paper:

```bibtex
@inproceedings{girish2025micunet,
  title={Towards Attribution of Generators and Emotional Manipulation in Cross-Lingual Synthetic Speech using Geometric Learning},
  author={Girish and Akhtar, Mohd Mujtaba and Sheth, Farhan and Singh, Muskaan},
  booktitle={Findings of the IJCNLP-AACL 2025},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md)

## Acknowledgments

- The Emo-Fake dataset creators
- Speech Foundation Model developers
- ACL/IJCNLP-AACL 2025 organizers
- All contributors and reviewers

## Related Work

- [Speech Foundation Models](https://huggingface.co/spaces/models)
- [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112)
- [Emotional Speech Recognition](https://github.com/search?q=emotion+recognition+speech)

---

<div align="center">

**Built with ❤️ for detecting emotional manipulation in synthetic speech**

[⬆ back to top](#micunet-multi-curvature-learning-for-emotional-manipulation-detection-in-synthetic-speech)

</div>
