# Bone Abnormality Detection with Deep Learning

## Overview

This project implements a deep learning pipeline for detecting abnormalities in musculoskeletal X-ray images using the MURA dataset. The task is a binary classification problem where each image is classified as either normal or abnormal.

A pretrained convolutional neural network (ResNet18) is fine-tuned on a subset of the dataset to establish a baseline model. The project is designed to be extensible, with planned additions including interpretability methods (Grad-CAM), robustness analysis, and model comparisons.


## Dataset

This project uses the MURA (Musculoskeletal Radiographs) dataset.

- ~40,000 X-ray images  
- 7 body parts (wrist, elbow, shoulder, etc.)  
- Labels: normal (0) or abnormal (1)

The dataset should be placed in:

MURA-v1.1/

## Setup

### 1. Create environment
conda create -n mura_env python=3.10 -y
conda activate mura_env

### 2. Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn pillow

## How to Run

### Train the model
python train.py

- Trains a ResNet18 model  
- Prints loss and validation accuracy per epoch  
- Saves model to model.pth  

### Evaluate the model
python eval.py

- Loads trained model  
- Computes accuracy, ROC-AUC, confusion matrix  
- Displays plots  

## Key Implementation Details

- Model: Pretrained ResNet18 (fine-tuned)  
- Loss: BCEWithLogitsLoss  
- Optimizer: Adam  
- Input size: 224 × 224  
- Dataset subset used for faster experimentation (configurable)  

## Custom Code Contributions

- Dataset loader for MURA structure  
- Training and evaluation loops  
- Model adaptation for binary classification  
- Metric computation and visualization  

Reused libraries:
- PyTorch
- torchvision

## Current Results

- Validation Accuracy: ~0.60–0.64  
- Loss decreases steadily across epochs  
- Mild overfitting observed due to limited dataset size  

## Future Work

- Grad-CAM interpretability  
- Robustness testing (noise, blur, contrast changes)  
- Model comparison (ResNet vs DenseNet)  
- Training on larger dataset subsets  

## Notes

- Dataset not included due to size  
- CPU training may be slow  
- GPU recommended for full-scale experiments
