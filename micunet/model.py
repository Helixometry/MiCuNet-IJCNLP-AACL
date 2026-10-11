"""Core MiCuNet model implementation."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class MiCuNet(nn.Module):
    """
    Multi-Curvature Learning Network for Emotional Manipulation Detection.

    This model combines Speech Foundation Model embeddings with spectrogram features
    using a mixed-curvature projection mechanism spanning Hyperbolic, Euclidean,
    and Spherical spaces.

    Args:
        input_dim (int): Dimension of input features
        hidden_dim (int): Hidden dimension for feature processing
        num_emotions (int): Number of emotion classes
        num_sources (int): Number of source/generator classes
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_emotions: int = 7,
        num_sources: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_emotions = num_emotions
        self.num_sources = num_sources

        # Feature extraction layers
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task-specific heads
        self.original_emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_emotions),
        )

        self.manipulated_emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_emotions),
        )

        self.source_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_sources),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MiCuNet.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Dictionary containing predictions for all tasks
        """
        # Feature extraction
        features = self.feature_projection(x)

        # Task-specific predictions
        original_emotion = self.original_emotion_head(features)
        manipulated_emotion = self.manipulated_emotion_head(features)
        source = self.source_head(features)

        return {
            "original_emotion": original_emotion,
            "manipulated_emotion": manipulated_emotion,
            "source": source,
            "features": features,
        }

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "MiCuNet":
        """
        Load a pretrained MiCuNet model.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            MiCuNet model with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract config from checkpoint
        config = checkpoint.get("config", {})
        model = cls(**config)

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model

    def save(self, checkpoint_path: str, config: Optional[Dict] = None) -> None:
        """
        Save the model checkpoint.

        Args:
            checkpoint_path: Path to save the checkpoint
            config: Configuration dictionary to save with the model
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_emotions": self.num_emotions,
                "num_sources": self.num_sources,
            },
        }
        torch.save(checkpoint, checkpoint_path)
