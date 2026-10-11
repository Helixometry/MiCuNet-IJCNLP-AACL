"""Mixed-curvature projection layers for MiCuNet."""

import torch
import torch.nn as nn
import math
from typing import Optional


class MixedCurvatureProjection(nn.Module):
    """
    Mixed-curvature projection mechanism spanning Hyperbolic, Euclidean, and Spherical spaces.

    This layer enables the model to capture geometric relationships in emotional and
    spectral features across different curvature spaces.

    Args:
        input_dim (int): Input feature dimension
        output_dim (int): Output feature dimension
        curvature_weight (float): Weight parameter for curvature balancing
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        curvature_weight: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.curvature_weight = curvature_weight

        # Euclidean projection
        self.euclidean_proj = nn.Linear(input_dim, output_dim)

        # Hyperbolic projection (uses non-linear transformation)
        self.hyperbolic_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
        )

        # Spherical projection (normalized)
        self.spherical_proj = nn.Linear(input_dim, output_dim)

        # Curvature weights
        self.register_parameter(
            "curvature_alpha",
            nn.Parameter(torch.tensor([1.0 / 3.0]))
        )
        self.register_parameter(
            "curvature_beta",
            nn.Parameter(torch.tensor([1.0 / 3.0]))
        )
        self.register_parameter(
            "curvature_gamma",
            nn.Parameter(torch.tensor([1.0 / 3.0]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mixed-curvature projection.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Projected tensor of shape (batch_size, output_dim)
        """
        # Get projections in different curvature spaces
        euclidean = self.euclidean_proj(x)
        hyperbolic = self.hyperbolic_proj(x)
        spherical = self.spherical_proj(x)
        spherical = torch.nn.functional.normalize(spherical, dim=-1)

        # Combine projections with learnable weights
        output = (
            self.curvature_alpha * euclidean +
            self.curvature_beta * hyperbolic +
            self.curvature_gamma * spherical
        )

        return output


class TemporalGating(nn.Module):
    """
    Learnable temporal gating mechanism for adaptive feature weighting.

    Applies attention-based gating to temporal features, enabling the model to
    focus on relevant time steps for emotion detection.

    Args:
        feature_dim (int): Dimension of input features
        gating_dim (int): Hidden dimension for gating mechanism
    """

    def __init__(self, feature_dim: int, gating_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        # Gating mechanism
        self.gate_fc1 = nn.Linear(feature_dim, gating_dim)
        self.gate_fc2 = nn.Linear(gating_dim, feature_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply temporal gating to features.

        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim) or (batch_size, feature_dim)
            mask: Optional attention mask

        Returns:
            Gated features with same shape as input
        """
        # Compute gating weights
        gate = self.relu(self.gate_fc1(x))
        gate = self.sigmoid(self.gate_fc2(gate))

        # Apply mask if provided
        if mask is not None:
            gate = gate * mask.unsqueeze(-1)

        # Apply gating
        output = x * gate

        return output


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head attention mechanism for temporal feature aggregation.

    Args:
        feature_dim (int): Dimension of input features
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.head_dim = feature_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.fc_out = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-head attention forward pass.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Attention mask

        Returns:
            Attention output
        """
        batch_size = query.shape[0]

        # Linear projections
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        out = torch.matmul(attention, V)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.feature_dim)

        # Final linear projection
        out = self.fc_out(out)

        return out
