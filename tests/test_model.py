"""Unit tests for MiCuNet model."""

import unittest
import torch
from micunet.model import MiCuNet


class TestMiCuNet(unittest.TestCase):
    """Test cases for MiCuNet model."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MiCuNet(
            input_dim=768,
            hidden_dim=512,
            num_emotions=7,
            num_sources=3,
            dropout=0.3,
        )
        self.batch_size = 4
        self.input_dim = 768

    def test_model_initialization(self):
        """Test model can be initialized."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.input_dim, 768)
        self.assertEqual(self.model.hidden_dim, 512)

    def test_forward_pass(self):
        """Test forward pass."""
        x = torch.randn(self.batch_size, self.input_dim)
        output = self.model(x)

        self.assertIn('original_emotion', output)
        self.assertIn('manipulated_emotion', output)
        self.assertIn('source', output)
        self.assertIn('features', output)

        self.assertEqual(
            output['original_emotion'].shape,
            (self.batch_size, 7)
        )
        self.assertEqual(
            output['manipulated_emotion'].shape,
            (self.batch_size, 7)
        )
        self.assertEqual(
            output['source'].shape,
            (self.batch_size, 3)
        )

    def test_model_save_load(self):
        """Test model save and load."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')

            # Save model
            self.model.save(checkpoint_path)
            self.assertTrue(os.path.exists(checkpoint_path))

            # Load model
            loaded_model = MiCuNet.from_pretrained(checkpoint_path)
            self.assertIsNotNone(loaded_model)

            # Test loaded model
            x = torch.randn(self.batch_size, self.input_dim)
            output = loaded_model(x)
            self.assertIn('original_emotion', output)


class TestMiCuNetGradients(unittest.TestCase):
    """Test gradient flow in MiCuNet."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MiCuNet(
            input_dim=768,
            hidden_dim=512,
            num_emotions=7,
            num_sources=3,
        )

    def test_backward_pass(self):
        """Test backward pass."""
        x = torch.randn(4, 768, requires_grad=True)
        output = self.model(x)

        # Create loss
        loss = output['original_emotion'].sum()
        loss.backward()

        # Check gradients
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


if __name__ == '__main__':
    unittest.main()
