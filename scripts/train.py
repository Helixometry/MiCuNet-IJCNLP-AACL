"""Training script for MiCuNet."""

import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path
import yaml

from micunet.model import MiCuNet
from data.loaders import AudioDataLoader
from micunet.utils import seed_everything, get_device, AverageMeter


logger = logging.getLogger(__name__)


class Trainer:
    """Training manager for MiCuNet."""

    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Initialize model
        self.model = MiCuNet(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_emotions=config['model']['num_emotions'],
            num_sources=config['model']['num_sources'],
            dropout=config['model']['dropout'],
        ).to(device)

        # Loss functions
        self.emotion_loss = nn.CrossEntropyLoss()
        self.source_loss = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )

    def train_epoch(self, train_loader, epoch):
        """Train one epoch."""
        self.model.train()

        loss_meter = AverageMeter()

        for batch_idx, batch in enumerate(train_loader):
            # Get data
            audio = batch['audio'].to(self.device)

            # Get targets if available
            targets = {}
            if 'original_emotion' in batch:
                targets['original_emotion'] = batch['original_emotion'].to(self.device)
            if 'manipulated_emotion' in batch:
                targets['manipulated_emotion'] = batch['manipulated_emotion'].to(self.device)
            if 'source' in batch:
                targets['source'] = batch['source'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(audio)

            # Compute loss
            loss = 0
            if 'original_emotion' in targets:
                loss += self.emotion_loss(
                    outputs['original_emotion'],
                    targets['original_emotion']
                ) / 3
            if 'manipulated_emotion' in targets:
                loss += self.emotion_loss(
                    outputs['manipulated_emotion'],
                    targets['manipulated_emotion']
                ) / 3
            if 'source' in targets:
                loss += self.source_loss(
                    outputs['source'],
                    targets['source']
                ) / 3

            # Backward pass
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), audio.size(0))

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss_meter.avg:.4f}"
                )

        return loss_meter.avg

    def train(self, train_loader, val_loader, num_epochs, output_dir):
        """Train model."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        best_loss = float('inf')

        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validation (simplified)
            val_loss = train_loss  # Placeholder

            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )

            # Save checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint_path = output_dir / f"best_checkpoint.pt"
                self.model.save(checkpoint_path, self.config['model'])
                logger.info(f"Saved best model to {checkpoint_path}")

            # Learning rate scheduling
            self.scheduler.step(val_loss)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train MiCuNet model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate

    # Setup
    seed_everything(config.get('seed', 42))
    device = get_device()
    logger.info(f"Using device: {device}")

    # Data loading
    train_loader = AudioDataLoader(
        config['data']['train_dir'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
    )

    val_loader = AudioDataLoader(
        config['data']['val_dir'],
        batch_size=config['training']['batch_size'],
        shuffle=False,
    )

    # Train
    trainer = Trainer(config, device)
    trainer.train(
        train_loader,
        val_loader,
        config['training']['epochs'],
        args.output_dir,
    )


if __name__ == "__main__":
    main()
