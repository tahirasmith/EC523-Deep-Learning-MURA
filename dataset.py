import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MURADataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        root_dir: path to MURA-v1.1/
        split: 'train' or 'valid'
        """
        self.samples = []
        self.transform = transform

        split_dir = os.path.join(root_dir, split)

        # Walk through directory
        for root, dirs, files in os.walk(split_dir):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)

                    # Label is in folder name: 'positive' or 'negative'
                    if "positive" in img_path:
                        label = 1
                    elif "negative" in img_path:
                        label = 0
                    else:
                        continue

                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(root_dir, batch_size=32, num_workers=2):
    """
    Returns train and validation DataLoaders
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_dataset = MURADataset(root_dir, split="train", transform=transform)
    val_dataset = MURADataset(root_dir, split="valid", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
