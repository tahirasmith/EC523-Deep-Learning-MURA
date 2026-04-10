import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MURADataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, max_samples=None):
        self.root_dir = os.path.join(root_dir, split)

        self.samples = []

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])

        # Collect all samples
        for root, _, files in os.walk(self.root_dir):
            for f in files:
                if f.endswith(".png"):
                    path = os.path.join(root, f)

                    if "positive" in path:
                        label = 1
                    elif "negative" in path:
                        label = 0
                    else:
                        continue

                    self.samples.append((path, label))

        # adding this because computer overheats
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label


def get_dataloaders(root_dir="MURA-v1.1", batch_size=4, max_samples=None):
    train_dataset = MURADataset(root_dir, split="train", max_samples=max_samples)
    val_dataset = MURADataset(root_dir, split="valid", max_samples=max_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader
