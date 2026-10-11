"""Utility functions for MiCuNet."""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Union, Optional, Tuple, Dict


def load_audio(
    audio_path: Union[str, Path],
    sr: int = 16000,
    mono: bool = True,
) -> torch.Tensor:
    """
    Load audio file using librosa.

    Args:
        audio_path: Path to audio file
        sr: Sample rate
        mono: Convert to mono

    Returns:
        Audio tensor of shape (samples,) or (channels, samples)
    """
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=mono)
    return torch.from_numpy(audio).float()


def extract_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> torch.Tensor:
    """
    Extract mel-spectrogram from audio.

    Args:
        audio: Audio tensor
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bands

    Returns:
        Mel-spectrogram of shape (n_mels, time_steps)
    """
    # Convert to numpy if needed
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()

    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    # Convert to log scale
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return torch.from_numpy(mel_spec).float()


def normalize_features(
    features: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize features using z-score normalization.

    Args:
        features: Feature tensor
        mean: Precomputed mean (optional)
        std: Precomputed std (optional)

    Returns:
        Normalized features, mean, std
    """
    if mean is None:
        mean = features.mean(dim=0)
    if std is None:
        std = features.std(dim=0) + 1e-8

    normalized = (features - mean) / std

    return normalized, mean, std


def get_device() -> torch.device:
    """Get the best available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    checkpoint_path: Union[str, Path],
    config: Optional[Dict] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        checkpoint_path: Path to save checkpoint
        config: Configuration dictionary
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "config": config,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load onto

    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


class AverageMeter:
    """Keep track of average value."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset the meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter.

        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
