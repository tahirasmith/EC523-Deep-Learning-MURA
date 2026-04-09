Bone Abnormality Detection with Deep Learning
Overview

This project implements a deep learning pipeline for detecting abnormalities in musculoskeletal X-ray images using the MURA dataset. The model classifies images as normal or abnormal using convolutional neural networks (CNNs).

The primary goal is to build a strong baseline model, with potential extensions including interpretability (e.g., Grad-CAM) and analysis under image perturbations.

Project Structure:
project/
│
├── data/                # MURA dataset (not included in repo)
├── models/              # Saved model weights
│
├── dataset.py           # Data loading and preprocessing
├── model.py             # Model architectures (ResNet, DenseNet)
├── train.py             # Training pipeline
├── eval.py              # Evaluation + visualization
│
├── requirements.txt
└── README.md

File Descriptions
dataset.py
Loads the MURA dataset from disk
Assigns labels (normal = 0, abnormal = 1)
Applies preprocessing (resize, normalization)
Returns PyTorch DataLoaders for training and validation

model.py
Defines models (ResNet18, DenseNet121)
Uses pretrained ImageNet weights
Replaces final layer for binary classification

train.py
Trains the model on the training dataset
Uses BCEWithLogitsLoss for binary classification
Evaluates on validation set after each epoch
Prints loss and validation accuracy

eval.py
Loads a trained model
Evaluates on validation data
Computes:
Accuracy
ROC-AUC
Generates:
ROC curve
Confusion matrix

Setup:

1. Install dependencies
pip install -r requirements.txt

2. Download the dataset
Download the MURA dataset and place it in:
data/MURA-v1.1/
Update the dataset path in train.py and eval.py if needed.

How to Run:
Train the model
python train.py
This will:
Train a CNN (default: ResNet18)
Output training loss and validation accuracy

Evaluate the model:
python eval.py

This will:
Load the trained model
Print accuracy and ROC-AUC
Display:
ROC curve
Confusion matrix






