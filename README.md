# Bone Abnormality Detection with Deep Learning

## Overview

This project implements a deep learning pipeline for detecting abnormalities in musculoskeletal X-ray images using the MURA dataset. The task is a binary classification problem where each image is classified as either normal or abnormal.

Two convolutional neural network architectures are explored:
- ResNet18 (baseline model)
- DenseNet121 (improved model)

The project includes training, evaluation, per-body-part analysis, and interpretability using Grad-CAM.

---

## Dataset

This project uses the MURA (Musculoskeletal Radiographs) dataset.

- ~40,000 X-ray images
- 7 body parts (hand, wrist, elbow, shoulder, finger, forearm, humerus)
- Labels: normal (0) or abnormal (1)

The dataset is pre-split into:
- train/
- valid/

---

## Setup

### 1. Create environment
conda create -n mura_env python=3.10 -y
conda activate mura_env

### 2. Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn pillow

---

## How to Run

### Train the ResNet model
python train.py

- Trains a ResNet18 model
- Prints loss and validation accuracy per epoch
- Saves model to model.pth

---

### Train the DenseNet model (recommended)
python train_densenet.py

- Trains a DenseNet121 model
- Uses balanced sampling
- Saves best model to densenet_best.pth

---

### Evaluate the model
python test.py

- Computes ROC-AUC
- Displays confusion matrix
- Computes Cohen’s Kappa
- Reports false positives and false negatives

---

### Per-body-part evaluation
python eval_by_bodypart.py

- Computes ROC-AUC for each body part

---

### Grad-CAM visualization
python gradcam_eval.py

- Generates Grad-CAM heatmaps
- Saves original, heatmap, and overlay images

---

## Key Implementation Details

- Model:
  - ResNet18 (baseline)
  - DenseNet121 (final model)

- Loss: BCEWithLogitsLoss

- Optimizer: Adam

- Input size: 224 × 224

- Sampling: balanced sampling to handle class imbalance

---

## Custom Code Contributions

- Dataset loader for MURA structure
- Balanced sampling implementation
- Training and evaluation loops
- Per-body-part evaluation metrics
- Confusion matrix and Cohen’s Kappa computation
- Grad-CAM visualization pipeline

Reused libraries:
- PyTorch
- torchvision

---

## Evaluation Metrics

- ROC-AUC: measures how well the model separates normal vs abnormal images
- Confusion Matrix: shows true/false positives and negatives
- Cohen’s Kappa: measures agreement adjusted for chance

---

## Results

### DenseNet121 (final model)

- ROC-AUC: ~0.78
- Cohen’s Kappa: ~0.44

Confusion Matrix:
            Pred 0   Pred 1
True 0      1245     422
True 1      469      1061

---

### ResNet18 (baseline model)

Confusion Matrix:
            Pred 0   Pred 1
True 0      1371     502
True 1      389      945

---

## Key Observations

- DenseNet achieves better overall balance between false positives and false negatives
- ResNet reduces false negatives (fewer missed abnormalities) but increases false positives
- Performance varies significantly across body parts (e.g., shoulder is more challenging)
- Model performance is stronger in smaller, clearer structures (hand, finger)

---

## Future Work

- Implement study-level prediction (aggregate multiple images per study)
- Improve class weighting (per body part, similar to original paper)
- Tune decision threshold to improve Cohen’s Kappa
- Expand training dataset usage
- Improve Grad-CAM stability and visualization quality
