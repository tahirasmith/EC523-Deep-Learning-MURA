import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

from dataset import get_dataloaders
from model import get_model


def evaluate():
    # ------------------
    # Config
    # ------------------
    data_path = "./data/MURA-v1.1"  # same as train.py
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------
    # Load Data
    # ------------------
    _, val_loader = get_dataloaders(data_path, batch_size=batch_size)

    # ------------------
    # Load Model
    # ------------------
    model = get_model("resnet18")
    model.load_state_dict(torch.load("models/model.pth", map_location=device))  # optional if saved
    model = model.to(device)
    model.eval()

    all_labels = []
    all_probs = []

    # ------------------
    # Inference
    # ------------------
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ------------------
    # Metrics
    # ------------------
    preds = (all_probs > 0.5).astype(int)
    accuracy = (preds == all_labels).mean()
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    # ------------------
    # ROC Curve
    # ------------------
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid()
    plt.show()

    # ------------------
    # Confusion Matrix
    # ------------------
    cm = confusion_matrix(all_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    evaluate()
