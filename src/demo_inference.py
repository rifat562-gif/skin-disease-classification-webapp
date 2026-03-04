#!/usr/bin/env python3
"""
demo_inference.py (SAFE DEMO)

Inference-only demo for the Skin Disease Classification Web App.
- No dataset required
- No trained model weights included

What it does:
1) Loads an image from disk
2) Applies basic preprocessing (resize + normalize)
3) Produces a placeholder prediction (since weights are not shipped)

If you later add your trained model file (e.g., models/model.h5),
this script can be upgraded to run real predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# Update these to match your project settings
IMG_SIZE = (128, 128)   # (width, height)
CLASS_NAMES = [
    "akiec",  # Actinic keratoses
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevi
    "vasc",   # Vascular lesions
]


def preprocess_image(image_path: Path) -> np.ndarray:
    """Load image -> RGB -> resize -> normalize -> add batch dimension."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [0,1]
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


def placeholder_predict(x: np.ndarray) -> tuple[str, float]:
    """Placeholder prediction (no model weights shipped)."""
    score = float(x.mean())
    idx = int((score * 1000) % len(CLASS_NAMES))
    confidence = 0.50 + 0.50 * float((score * 10) % 1.0)  # 0.50–1.00
    return CLASS_NAMES[idx], round(confidence, 4)


def main():
    parser = argparse.ArgumentParser(description="Inference-only demo (no model weights).")
    parser.add_argument("--image", "-i", required=True, help="Path to input image (jpg/png).")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    x = preprocess_image(image_path)
    label, conf = placeholder_predict(x)

    print("\nDemo Inference (placeholder)")
    print(f"Image      : {image_path.name}")
    print(f"Input shape: {x.shape}")
    print(f"Prediction : {label}")
    print(f"Confidence : {conf}")
    print("\nNote: This is a placeholder prediction because trained weights are not included.")


if __name__ == "__main__":
    main()
