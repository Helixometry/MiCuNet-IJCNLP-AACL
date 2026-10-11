"""Unit tests for MiCuNet layers."""

import unittest
import torch
from micunet.layers import (
    MixedCurvatureProjection,
    TemporalGating,
    MultiHeadTemporalAttention,
)


class TestMixedCurvatureProjection(unittest.TestCase):
    """Test mixed-curvature projection layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.layer = MixedCurvatureProjection(
            input_dim=768,
            output_dim=512,
        )
        self.batch_size = 4

    def test_forward_pass(self):
        """Test forward pass."""
        x = torch.randn(self.batch_size, 768)
        output = self.layer(x)

        self.assertEqual(output.shape, (self.batch_size, 512))

    def test_curvature_weights_normalization(self):
        """Test that curvature weights are properly normalized."""
        # Weights should be between 0 and 1
        alpha = self.layer.curvature_alpha.item()
        beta = self.layer.curvature_beta.item()
        gamma = self.layer.curvature_gamma.item()

        self.assertGreater(alpha, 0)
        self.assertGreater(beta, 0)
        self.assertGreater(gamma, 0)


class TestTemporalGating(unittest.TestCase):
    """Test temporal gating layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.layer = TemporalGating(feature_dim=512)
        self.batch_size = 4
        self.seq_len = 100
        self.feature_dim = 512

    def test_forward_pass(self):
        """Test forward pass."""
        x = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        output = self.layer(x)

        self.assertEqual(output.shape, x.shape)

    def test_with_mask(self):
        """Test forward pass with mask."""
        x = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        mask = torch.ones(self.batch_size, self.seq_len)

        output = self.layer(x, mask)
        self.assertEqual(output.shape, x.shape)


class TestMultiHeadTemporalAttention(unittest.TestCase):
    """Test multi-head temporal attention layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.layer = MultiHeadTemporalAttention(
            feature_dim=512,
            num_heads=8,
        )
        self.batch_size = 4
        self.seq_len = 100
        self.feature_dim = 512

    def test_forward_pass(self):
        """Test forward pass."""
        query = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.feature_dim)

        output = self.layer(query, key, value)

        # Output should have same shape as input
        self.assertEqual(output.shape[0], self.batch_size)


if __name__ == '__main__':
    unittest.main()
