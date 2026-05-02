import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import config


class Evaluator:
    def __init__(self, model: nn.Module, test_loader: DataLoader):
        self.model       = model.to(config.DEVICE)
        self.test_loader = test_loader

    def evaluate(self) -> tuple[float, np.ndarray]:
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(config.DEVICE)
                preds  = self.model(images).argmax(1).cpu()
                all_preds.append(preds)
                all_labels.append(labels)

        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        accuracy         = (all_preds == all_labels).mean()
        confusion_matrix = self._confusion_matrix(all_labels, all_preds)

        print(f"\nTest Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")
        self._print_per_class(all_labels, all_preds)

        return accuracy, confusion_matrix

    def _confusion_matrix(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        matrix = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=int)
        for true, pred in zip(labels, preds):
            matrix[true][pred] += 1
        return matrix

    def _print_per_class(self, labels: np.ndarray, preds: np.ndarray) -> None:
        print(f"\n{'Class':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
        print("-" * 40)
        for i, cls in enumerate(config.CIFAR10_CLASSES):
            mask    = labels == i
            correct = (preds[mask] == i).sum()
            total   = mask.sum()
            acc     = correct / total if total > 0 else 0.0
            print(f"{cls:<12} {correct:>8} {total:>8} {acc:>8.4f}")
