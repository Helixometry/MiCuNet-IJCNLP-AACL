"""Inference script for MiCuNet."""

import argparse
import torch
from pathlib import Path

from micunet.model import MiCuNet
from data.preprocessor import AudioPreprocessor


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


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Inference with MiCuNet")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Language"
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MiCuNet.from_pretrained(args.checkpoint)
    model = model.to(device)
    model.eval()

    # Load and preprocess audio
    preprocessor = AudioPreprocessor()
    audio = preprocessor.load_audio(args.audio)
    melspec = preprocessor.extract_melspectrogram(audio)

    # Convert to tensor
    melspec = torch.from_numpy(melspec).float().unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(melspec)

    # Extract predictions
    original_emotion_logits = outputs['original_emotion'][0]
    manipulated_emotion_logits = outputs['manipulated_emotion'][0]
    source_logits = outputs['source'][0]

    original_emotion_idx = original_emotion_logits.argmax().item()
    manipulated_emotion_idx = manipulated_emotion_logits.argmax().item()
    source_idx = source_logits.argmax().item()

    # Get confidence scores
    original_confidence = torch.softmax(original_emotion_logits, dim=0)[original_emotion_idx].item()
    manipulated_confidence = torch.softmax(manipulated_emotion_logits, dim=0)[manipulated_emotion_idx].item()
    source_confidence = torch.softmax(source_logits, dim=0)[source_idx].item()

    # Print results
    print(f"\n{'='*60}")
    print(f"Inference Results for: {Path(args.audio).name}")
    print(f"Language: {args.language}")
    print(f"{'='*60}")

    print(f"\nOriginal Emotion:")
    print(f"  Class: {EMOTION_LABELS[original_emotion_idx]}")
    print(f"  Confidence: {original_confidence:.4f}")

    print(f"\nManipulated Emotion:")
    print(f"  Class: {EMOTION_LABELS[manipulated_emotion_idx]}")
    print(f"  Confidence: {manipulated_confidence:.4f}")

    print(f"\nSource/Generator:")
    print(f"  Class: {SOURCE_LABELS[source_idx]}")
    print(f"  Confidence: {source_confidence:.4f}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
