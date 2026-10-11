"""Feature encoders for MiCuNet."""

import torch
import torch.nn as nn
from typing import Optional


class SpeechFoundationModelEncoder(nn.Module):
    """
    Encoder for speech foundation model embeddings.

    This encoder processes semantic-prosodic embeddings from speech foundation models.

    Args:
        input_dim (int): Input embedding dimension
        output_dim (int): Output feature dimension
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode speech foundation model embeddings.

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            Encoded features of shape (batch_size, output_dim)
        """
        return self.encoder(x)


class SpectrogramEncoder(nn.Module):
    """
    Encoder for spectrogram features.

    Processes spectrogram features using convolutional layers to extract
    spectral and temporal patterns.

    Args:
        n_mels (int): Number of mel bands
        output_dim (int): Output feature dimension
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        n_mels: int = 128,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.output_dim = output_dim

        # Convolutional layers for spectrogram processing
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Dense layers
        self.fc = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode spectrogram features.

        Args:
            x: Spectrogram of shape (batch_size, n_mels, time_steps)

        Returns:
            Encoded features of shape (batch_size, output_dim)
        """
        # Convolutional processing
        x = self.conv1(x)
        x = self.conv2(x)

        # Global average pooling
        x = self.pool(x)
        x = x.squeeze(-1)

        # Dense layers
        x = self.fc(x)

        return x


class DualStreamEncoder(nn.Module):
    """
    Dual-stream encoder combining speech foundation model and spectrogram features.

    Args:
        foundation_model_dim (int): Dimension of foundation model embeddings
        n_mels (int): Number of mel bands in spectrogram
        output_dim (int): Output feature dimension
        dropout (float): Dropout probability
        fusion_type (str): Feature fusion method ('concat' or 'add')
    """

    def __init__(
        self,
        foundation_model_dim: int = 1024,
        n_mels: int = 128,
        output_dim: int = 512,
        dropout: float = 0.1,
        fusion_type: str = "concat",
    ):
        super().__init__()
        self.fusion_type = fusion_type

        # Foundation model encoder
        self.fm_encoder = SpeechFoundationModelEncoder(
            input_dim=foundation_model_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

        # Spectrogram encoder
        self.spec_encoder = SpectrogramEncoder(
            n_mels=n_mels,
            output_dim=output_dim,
            dropout=dropout,
        )

        # Fusion layer
        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif fusion_type == "add":
            self.fusion = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(
        self,
        foundation_embeddings: torch.Tensor,
        spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode both streams and fuse features.

        Args:
            foundation_embeddings: Foundation model embeddings
            spectrogram: Spectrogram features

        Returns:
            Fused features of shape (batch_size, output_dim)
        """
        # Encode each stream
        fm_features = self.fm_encoder(foundation_embeddings)
        spec_features = self.spec_encoder(spectrogram)

        # Fuse features
        if self.fusion_type == "concat":
            fused = torch.cat([fm_features, spec_features], dim=-1)
        elif self.fusion_type == "add":
            fused = fm_features + spec_features

        # Apply fusion layer
        output = self.fusion(fused)

        return output
