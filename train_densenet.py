import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from torchvision import models


def get_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = get_dataloaders(batch_size=8, max_samples=10000)

    # Model
    model = get_model().to(device)

    
    pos = 0
    neg = 0
    for _, labels in train_loader:
        labels = labels.view(-1)
        pos += (labels == 1).sum().item()
        neg += (labels == 0).sum().item()

    print(f"Train positives: {pos}, negatives: {neg}")

    pos_weight = torch.tensor([neg / pos], dtype=torch.float).to(device) # hoping this makes up for class imbalance of pos to negative
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    epochs = 25

    best_val_acc = 0.0

    for epoch in range(epochs):
        # train
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

        # validation
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

        val_acc = correct / total

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

        # save best model, see at what epoch does model stop learning. use this to add early stopping if necessary
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "densenet_best.pth")
            print("Saved best model")

    print("Training complete")


if __name__ == "__main__":
    train()
