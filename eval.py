import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix

from dataset import get_dataloaders
from model import get_model


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = get_dataloaders()

    model = get_model().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    auc = roc_auc_score(all_labels, all_probs)
    print("ROC-AUC:", auc)

    preds = (all_probs > 0.5).astype(int)
    cm = confusion_matrix(all_labels, preds)
    print("Confusion Matrix:\n", cm)

    plt.figure()
    plt.hist(all_probs, bins=20)
    plt.title("Prediction Distribution")
    plt.show()
