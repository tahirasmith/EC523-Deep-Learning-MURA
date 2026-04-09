Bone Abnormality Detection with Deep Learning
Overview

This project implements a deep learning pipeline for detecting abnormalities in musculoskeletal X-ray images using the MURA dataset. The goal is to train a convolutional neural network (CNN) to classify images as normal or abnormal, and to evaluate model performance using standard metrics and visualizations.

The project is structured to first establish a strong baseline model, with potential extensions including interpretability methods (e.g., Grad-CAM) and robustness analysis.

Project Structure
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
Defines deep learning models (ResNet18, DenseNet121)
Uses pretrained weights (ImageNet)
Modifies final layer for binary classification
train.py
Trains the model on the training dataset
Uses binary cross-entropy loss (BCEWithLogitsLoss)
Evaluates on validation set after each epoch
Prints loss and accuracy
eval.py
Loads a trained model
Evaluates performance on validation set
Computes:
Accuracy
ROC-AUC
Generates visualizations:
ROC curve
Confusion matrix
Setup
1. Install dependencies
pip install -r requirements.txt
2. Download the dataset

Download the MURA dataset from Stanford and place it in:

data/MURA-v1.1/

Update the path in train.py and eval.py if needed.

How to Run
Train the model
python train.py

This will:

Train a CNN (default: ResNet18)
Print training loss and validation accuracy
Evaluate the model
python eval.py

This will:

Load the trained model
Print accuracy and ROC-AUC
Display:
ROC curve
Confusion matrix
