"""Audio preprocessing utilities."""

import numpy as np
import librosa
import torch
from typing import Union, Tuple, Optional


class AudioPreprocessor:
    """
    Audio preprocessing pipeline.

    Args:
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bands
    """

    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def load_audio(self, audio_path: str, mono: bool = True) -> np.ndarray:
        """Load audio file."""
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=mono)
        return audio

    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return mfcc

    def extract_melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec

    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features."""
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return chroma

    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """Extract zero crossing rate."""
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            hop_length=self.hop_length,
        )
        return zcr

    def normalize(
        self,
        features: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize features (z-score)."""
        if mean is None:
            mean = features.mean(axis=1, keepdims=True)
        if std is None:
            std = features.std(axis=1, keepdims=True) + 1e-8

        normalized = (features - mean) / std
        return normalized, mean, std

    def process(self, audio_path: str) -> dict:
        """
        Complete preprocessing pipeline.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with various features
        """
        audio = self.load_audio(audio_path)

        features = {
            'mfcc': self.extract_mfcc(audio),
            'melspectrogram': self.extract_melspectrogram(audio),
            'chroma': self.extract_chroma(audio),
            'zcr': self.extract_zero_crossing_rate(audio),
        }

        return features
