# Galaxy10 Image Classifier

A deep learning project that classifies galaxy images into 10 morphological types using transfer learning with EfficientNet-B0.

## What It Does

Takes images of galaxies and predicts their type from 10 categories:
- Barred Spiral, Unbarred Loose Spiral, Unbarred Tight Spiral
- Edge-on with Bulge, Edge-on without Bulge
- Round Smooth, In-between Round Smooth, Cigar Round Smooth
- Distributed, Merging

## How It Works

**Model:** EfficientNet-B0 pretrained on ImageNet, with early layers frozen and a custom classification head.

**Training Strategy:**
1. **Phase 1 (5 epochs):** Train only the classifier head while keeping the backbone frozen.
2. **Phase 2 (15 epochs):** Unfreeze the last 3 backbone layers and fine-tune end-to-end with a lower learning rate.

Both phases use AdamW optimizer with Cosine Annealing LR scheduling and label smoothing (0.1) in the loss.

**Data Augmentation:** Since galaxies are rotationally invariant, training uses aggressive augmentation — random flips, full 180° rotations, color jitter, and Gaussian noise.

**Stress Testing:** A separate "harsh" transform pipeline (heavy blur + JPEG compression + noise) is used to evaluate robustness.

## Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | 60% |
| Macro F1 | 0.58 |

Best performance on edge-on galaxy types (with buldge ~73 without buldge ~78% F1). Weakest on "Distributed" galaxies (~34% F1), likely due to their ambiguous appearance.

## Requirements

```
torch
torchvision
scikit-learn
matplotlib
seaborn
numpy
```

## Dataset

Uses the **Galaxy10** dataset, expected in:
```
raw/image_folders/train/galaxy10_train/
raw/image_folders/valid/galaxy10_test/
```
