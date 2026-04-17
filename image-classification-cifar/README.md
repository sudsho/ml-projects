# Image Classification with CNNs — CIFAR-10

Deep learning project building a CNN from scratch and applying transfer learning with ResNet-18 on the CIFAR-10 dataset.

## Overview

CIFAR-10 contains 60,000 32×32 color images across 10 classes (50k train / 10k test). This project explores two approaches:

1. **CNN from scratch** — custom architecture with Conv blocks, BatchNorm, and Dropout
2. **Transfer learning** — ResNet-18 pre-trained on ImageNet, fine-tuned on CIFAR-10

## Project Structure

```
image-classification-cifar/
├── 01_data_exploration.py        # Data loading, augmentation setup, visualization
├── 02_cnn_from_scratch.py        # Custom CNN architecture definition
├── 03_training_loop.py           # Training loop, learning rate scheduling, loss curves
├── 04_transfer_learning_resnet.py # ResNet-18 fine-tuning with frozen/unfrozen layers
├── 05_comparison_visualization.py # Confusion matrices, per-class accuracy, Grad-CAM
└── README.md
```

## Results

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| CNN from Scratch | ~82% | 3 ConvBlocks, BatchNorm, Dropout |
| ResNet-18 (Transfer) | ~92% | ImageNet pre-trained, fine-tuned 30 epochs |

Transfer learning provides ~10% accuracy improvement with faster convergence.

## Key Techniques

- **Data augmentation**: random horizontal flip, random crop with padding
- **Batch normalization**: stabilizes training, allows higher learning rates
- **Dropout**: regularization to prevent overfitting
- **Learning rate scheduling**: CosineAnnealingLR for smooth convergence
- **Transfer learning**: freeze early layers, fine-tune final blocks + classifier
- **Grad-CAM**: gradient-weighted class activation maps for model interpretability

## Visualizations

- `confusion_matrices.png` — normalized confusion matrix for both models
- `per_class_accuracy.png` — per-class breakdown (cat and dog hardest to classify)
- `model_comparison.png` — overall accuracy bar chart
- `gradcam_visualization.png` — Grad-CAM heatmaps showing what the model focuses on

## Setup

```bash
pip install torch torchvision matplotlib seaborn scikit-learn numpy
```

CIFAR-10 is downloaded automatically via `torchvision.datasets.CIFAR10`.

## Observations

- The model struggles most with **cat vs dog** and **deer vs horse** due to visual similarity
- Grad-CAM shows ResNet attends to animal faces and body shapes appropriately
- Transfer learning converges in ~10 epochs vs ~50 for training from scratch
