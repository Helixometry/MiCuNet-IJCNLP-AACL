"""Evaluation script for MiCuNet."""

import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

from micunet.model import MiCuNet
from data.loaders import AudioDataLoader


EMOTION_LABELS = [
    "Neutral",
    "Happiness",
    "Sadness",
    "Anger",
    "Surprise",
    "Fear",
    "Disgust",
]

SOURCE_LABELS = [
    "TTS Model A",
    "TTS Model B",
    "TTS Model C",
]


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()

    all_original_preds = []
    all_original_targets = []
    all_manipulated_preds = []
    all_manipulated_targets = []
    all_source_preds = []
    all_source_targets = []

    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio'].to(device)

            # Forward pass
            outputs = model(audio)

            # Get predictions
            original_preds = outputs['original_emotion'].argmax(dim=1)
            manipulated_preds = outputs['manipulated_emotion'].argmax(dim=1)
            source_preds = outputs['source'].argmax(dim=1)

            # Store predictions and targets
            if 'original_emotion' in batch:
                all_original_preds.extend(original_preds.cpu().numpy())
                all_original_targets.extend(batch['original_emotion'].numpy())

            if 'manipulated_emotion' in batch:
                all_manipulated_preds.extend(manipulated_preds.cpu().numpy())
                all_manipulated_targets.extend(batch['manipulated_emotion'].numpy())

            if 'source' in batch:
                all_source_preds.extend(source_preds.cpu().numpy())
                all_source_targets.extend(batch['source'].numpy())

    results = {
        'original_emotion': {
            'predictions': np.array(all_original_preds),
            'targets': np.array(all_original_targets),
        },
        'manipulated_emotion': {
            'predictions': np.array(all_manipulated_preds),
            'targets': np.array(all_manipulated_targets),
        },
        'source': {
            'predictions': np.array(all_source_preds),
            'targets': np.array(all_source_targets),
        },
    }

    return results


def print_results(results, output_dir=None):
    """Print evaluation results."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    for task_name, task_results in results.items():
        if len(task_results['predictions']) == 0:
            continue

        print(f"\n{task_name.upper()}")
        print("-"*80)

        # Get labels
        if 'emotion' in task_name:
            labels = EMOTION_LABELS
        else:
            labels = SOURCE_LABELS

        # Classification report
        report = classification_report(
            task_results['targets'],
            task_results['predictions'],
            target_names=labels,
            digits=4,
        )
        print(report)

        # Confusion matrix
        cm = confusion_matrix(
            task_results['targets'],
            task_results['predictions'],
        )
        print("Confusion Matrix:")
        print(cm)

        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / f"{task_name}_report.txt", 'w') as f:
                f.write(report)

            np.save(
                output_dir / f"{task_name}_confusion_matrix.npy",
                cm,
            )

    print("\n" + "="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate MiCuNet model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to test data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = MiCuNet.from_pretrained(args.checkpoint)
    model = model.to(device)

    # Load test data
    print(f"Loading test data from {args.test_dir}")
    test_loader = AudioDataLoader(
        args.test_dir,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Evaluate
    print("Evaluating...")
    results = evaluate(model, test_loader, device)

    # Print results
    print_results(results, args.output_dir)


if __name__ == "__main__":
    main()
