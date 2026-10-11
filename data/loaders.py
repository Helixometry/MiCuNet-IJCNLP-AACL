"""Data loading utilities for MiCuNet."""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np


class AudioDataset(Dataset):
    """
    PyTorch Dataset for audio files.

    Args:
        audio_dir: Directory containing audio files
        labels_file: CSV file with audio paths and labels
        sr: Sample rate
        duration: Audio duration in seconds (pads or truncates)
    """

    def __init__(
        self,
        audio_dir: Union[str, Path],
        labels_file: Optional[Union[str, Path]] = None,
        sr: int = 16000,
        duration: float = 3.0,
    ):
        self.audio_dir = Path(audio_dir)
        self.sr = sr
        self.max_length = int(sr * duration)

        # Load audio files
        if labels_file:
            self.audio_paths = self._load_from_labels(labels_file)
        else:
            self.audio_paths = list(self.audio_dir.glob("**/*.wav"))

    def _load_from_labels(self, labels_file: Union[str, Path]) -> List[Tuple]:
        """Load audio paths and labels from file."""
        import csv

        data = []
        with open(labels_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'path': self.audio_dir / row['audio_path'],
                    'original_emotion': int(row.get('original_emotion', 0)),
                    'manipulated_emotion': int(row.get('manipulated_emotion', 0)),
                    'source': int(row.get('source', 0)),
                })
        return data

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> dict:
        """Get audio sample."""
        import librosa

        if isinstance(self.audio_paths[0], dict):
            item = self.audio_paths[idx]
            audio_path = item['path']
            labels = {
                'original_emotion': item['original_emotion'],
                'manipulated_emotion': item['manipulated_emotion'],
                'source': item['source'],
            }
        else:
            audio_path = self.audio_paths[idx]
            labels = {}

        # Load audio
        audio, _ = librosa.load(str(audio_path), sr=self.sr, mono=True)

        # Pad or truncate
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        else:
            audio = audio[:self.max_length]

        return {
            'audio': torch.from_numpy(audio).float(),
            'path': str(audio_path),
            **labels,
        }


class AudioDataLoader:
    """Wrapper for AudioDataset with DataLoader."""

    def __init__(
        self,
        audio_dir: Union[str, Path],
        labels_file: Optional[Union[str, Path]] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        sr: int = 16000,
        duration: float = 3.0,
    ):
        self.dataset = AudioDataset(
            audio_dir=audio_dir,
            labels_file=labels_file,
            sr=sr,
            duration=duration,
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.loader)

    def __len__(self):
        """Get number of batches."""
        return len(self.loader)
