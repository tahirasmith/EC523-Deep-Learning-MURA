import torch
from sklearn.metrics import roc_auc_score, confusion_matrix

from dataset import get_dataloaders
from torchvision import models
import torch.nn as nn


def get_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    _, val_loader = get_dataloaders(batch_size=8, max_samples=10000)

    # Model
    model = get_model().to(device)
    model.load_state_dict(torch.load("densenet_best.pth", map_location=device))
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs).view(-1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    auc = roc_auc_score(all_labels, all_probs)
    preds = [1 if p > 0.5 else 0 for p in all_probs]  # use your better threshold

    cm = confusion_matrix(all_labels, preds)

    print(f"ROC-AUC: {auc}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    test()
