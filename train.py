import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import get_model


def train():
    # ------------------
    # Config
    # ------------------
    data_path = "./data/MURA-v1.1" ## come back and change this because i am swapping devices so path is diff
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # ------------------
    # Data
    # ------------------
    train_loader, val_loader = get_dataloaders(data_path, batch_size=batch_size)

    # ------------------
    # Model
    # ------------------
    model = get_model("resnet18")
    model = model.to(device)

    # ------------------
    # Loss + Optimizer
    # ------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------
    # Training loop
    # ------------------
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ------------------
        # Validation
        # ------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images).squeeze()
                preds = torch.sigmoid(outputs) > 0.5

                correct += (preds.int() == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {avg_train_loss:.4f} "
              f"Val Acc: {accuracy:.4f}")


if __name__ == "__main__":
    train()
