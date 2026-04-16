import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import get_model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(batch_size=8, max_samples=10000)

    model = get_model().to(device)

    # --------------------------
    # Compute class imbalance
    # --------------------------
    pos = 0
    neg = 0

    for _, labels in train_loader:
        labels = labels.view(-1)
        pos += (labels == 1).sum().item()
        neg += (labels == 0).sum().item()

    print(f"Train positives: {pos}, negatives: {neg}")

    # --------------------------
    # Weighted loss
    # --------------------------
    pos_weight = torch.tensor([neg / pos], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    epochs = 25

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # --------------------------
        # Validation
        # --------------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).cpu()

                correct += (preds.squeeze() == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={acc:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


if __name__ == "__main__":
    train()
