import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import config


class Evaluator:
    def __init__(self, model: nn.Module, test_loader: DataLoader):
        self.model       = model.to(config.DEVICE)
        self.test_loader = test_loader

    def evaluate(self):
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
        return accuracy, confusion_matrix

    def _confusion_matrix(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        matrix = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=int)
        for true, pred in zip(labels, preds):
            matrix[true][pred] += 1
        return matrix
