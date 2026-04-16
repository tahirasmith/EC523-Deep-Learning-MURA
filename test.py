import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix

from dataset import get_dataloaders
from model import get_model


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Make sure this matches your training path
    train_loader, test_loader = get_dataloaders("./MURA-v1.1", batch_size=4)

    model = get_model().to(device)

    model_path = "model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images).view(-1)
            probs = torch.sigmoid(outputs)

            preds = (probs > 0.6).int()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().flatten().tolist())
            all_preds.extend(preds.detach().cpu().flatten().tolist())

    # Metrics
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"ROC-AUC: {auc}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    test()
