# Methodology (Short)

## Goal
Real-time multi-class skin lesion classification from images using CNN-based deep learning, exposed through a web application.

## Dataset
- HAM10000 dermatoscopic image dataset (7 classes).
- Images are resized and normalized before training.

## Preprocessing
- Resize to a fixed input size (e.g., 128×128).
- Normalize pixel values to [0, 1].
- Train/validation split with class balancing strategy (if used).
- Data augmentation (rotation/flip/zoom) to improve generalization (if used).

## Models Evaluated
- Custom CNN (from scratch)
- ResNet50 (transfer learning)
- (Optional) MobileNetV2 / EfficientNet variants

## Training Setup (Summary)
- Loss: categorical cross-entropy
- Optimizer: Adam
- Metrics: accuracy (+ confusion matrix)
- Early stopping / learning-rate scheduling (if used)

## Evaluation
- Confusion matrix and per-class performance.
- Training curves (accuracy/loss) for comparison across models.

## Deployment (Web App)
- Flask API for inference (`/predict`)
- React frontend for image upload + displaying predicted label and confidence

## Disclaimer
This tool is for educational/research purposes only and does not replace professional medical advice.