import torch
import numpy as np
import os
import random

from collections import defaultdict
from sklearn.metrics import roc_auc_score, confusion_matrix, cohen_kappa_score

from dataset import get_dataloaders
from model_densenet import get_model
from torchvision.utils import save_image


def test(model_path="densenet_best.pth", threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = get_dataloaders(batch_size=4)

    model = get_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels = []
    all_probs = []

    false_pos = []
    false_neg = []

    # per-body-part tracking
    bp_true = defaultdict(list)
    bp_pred = defaultdict(list)

    with torch.no_grad():
        for images, labels, body_parts in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)
            probs = torch.sigmoid(outputs).view(-1)
            preds = (probs > threshold).float()

            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                prob = probs[i].item()
                bp = body_parts[i]

                all_labels.append(label)
                all_probs.append(prob)

                bp_true[bp].append(label)
                bp_pred[bp].append(pred)

                if label == 0 and pred == 1:
                    false_pos.append((images[i].cpu(), label, pred, prob))
                elif label == 1 and pred == 0:
                    false_neg.append((images[i].cpu(), label, pred, prob))

    # -------------------------
    # GLOBAL METRICS
    # -------------------------
    roc_auc = roc_auc_score(all_labels, all_probs)
    preds_binary = (np.array(all_probs) > threshold).astype(int)

    cm = confusion_matrix(all_labels, preds_binary)
    kappa = cohen_kappa_score(all_labels, preds_binary)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Cohen's Kappa: {kappa:.4f}")

    print(f"\nFalse Positives: {len(false_pos)}")
    print(f"False Negatives: {len(false_neg)}")

    # -------------------------
    # PER-BODY-PART KAPPA
    # -------------------------
    print("\n=== PER BODY PART COHEN'S KAPPA ===")
    for bp in bp_true:
        bp_kappa = cohen_kappa_score(bp_true[bp], bp_pred[bp])
        print(f"{bp}: Kappa={bp_kappa:.4f} | samples={len(bp_true[bp])}")

    # -------------------------
    # SAVE MISCLASSIFIED EXAMPLES
    # -------------------------
    random.shuffle(false_pos)
    random.shuffle(false_neg)

    os.makedirs("misclassified/false_pos", exist_ok=True)
    os.makedirs("misclassified/false_neg", exist_ok=True)

    for i, (img, label, pred, prob) in enumerate(false_pos[:20]):
        save_image(
            img,
            f"misclassified/false_pos/img_{i}_label{label}_pred{pred}_prob{prob:.2f}.png"
        )

    for i, (img, label, pred, prob) in enumerate(false_neg[:20]):
        save_image(
            img,
            f"misclassified/false_neg/img_{i}_label{label}_pred{pred}_prob{prob:.2f}.png"
        )

    print("\nSaved misclassified images.")


if __name__ == "__main__":
    test()
