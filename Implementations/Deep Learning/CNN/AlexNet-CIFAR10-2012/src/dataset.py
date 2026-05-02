import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import config


def _load_batch(path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        entry = pickle.load(f, encoding="bytes")
    images = entry[b"data"].reshape(-1, 3, 32, 32).astype(np.float32)
    labels = np.array(entry[b"labels"], dtype=np.int64)
    return images, labels


def _load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images, train_labels = [], []
    for i in range(1, 6):
        imgs, lbls = _load_batch(config.DATA_DIR / f"data_batch_{i}")
        train_images.append(imgs)
        train_labels.append(lbls)

    x_train = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    x_test, y_test = _load_batch(config.DATA_DIR / "test_batch")
    return x_train, y_train, x_test, y_test


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 dataset with optional augmentation.
    Images are stored as (N, 3, 32, 32) float32 in [0, 255].
    """

    _MEAN = (0.4914, 0.4822, 0.4465)
    _STD  = (0.2470, 0.2435, 0.2616)

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.Normalize(self._MEAN, self._STD),
            ])
        else:
            self.transform = transforms.Normalize(self._MEAN, self._STD)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx] / 255.0
        return self.transform(image), self.labels[idx]


def load_data() -> tuple[DataLoader, DataLoader, DataLoader]:
    x_train, y_train, x_test, y_test = _load_cifar10()

    full_train = CIFAR10Dataset(x_train, y_train, augment=True)
    test_set   = CIFAR10Dataset(x_test,  y_test,  augment=False)

    val_size   = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {train_size}  |  Val: {val_size}  |  Test: {len(test_set)}")
    return train_loader, val_loader, test_loader


def load_raw_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unnormalized arrays for classical ML baselines."""
    return _load_cifar10()
