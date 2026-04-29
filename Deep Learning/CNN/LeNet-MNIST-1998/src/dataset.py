import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import config


class MNISTDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_data():
    raw = np.load(config.DATA_PATH)
    x_train, y_train = raw["x_train"], raw["y_train"]
    x_test,  y_test  = raw["x_test"],  raw["y_test"]

    train_dataset = MNISTDataset(x_train, y_train)
    test_dataset  = MNISTDataset(x_test,  y_test)

    val_size   = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE)
    test_loader  = DataLoader(test_dataset,  batch_size=config.BATCH_SIZE)

    print(f"Train: {train_size}  |  Val: {val_size}  |  Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader
