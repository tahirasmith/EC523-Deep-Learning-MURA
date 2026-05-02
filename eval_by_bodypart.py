import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from gradcam import load_model, preprocess


def collect_by_bodypart(root):
    data = {}

    for r, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".png"):
                continue

            path = os.path.join(r, f)

            # label
            label = 1 if "positive" in path else 0 if "negative" in path else None
            if label is None:
                continue

            # body part extraction
            parts = path.split(os.sep)
            bodypart = None
            for p in parts:
                if p.startswith("XR_"):
                    bodypart = p
                    break

            if bodypart is None:
                continue

            if bodypart not in data:
                data[bodypart] = []

            data[bodypart].append((path, label))

    return data


def evaluate(model, samples):
    y_true = []
    y_score = []

    for path, label in samples:
        _, tensor = preprocess(path)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output).item()

        y_true.append(label)
        y_score.append(prob)

    return roc_auc_score(y_true, y_score)


if __name__ == "__main__":
    model = load_model("densenet_best.pth")

    root = "MURA-v1.1/train"
    data = collect_by_bodypart(root)

    print("\n=== PER BODY PART ROC-AUC ===")

    for bp, samples in data.items():
        if len(samples) < 50:
            continue

        auc = evaluate(model, samples)
        print(f"{bp}: AUC={auc:.4f} | samples={len(samples)}")

