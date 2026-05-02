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
            for X, y in self.test_loader:
                X     = X.to(config.DEVICE)
                preds = self.model(X).argmax(1).cpu()
                all_preds.append(preds)
                all_labels.append(y)

        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        accuracy = (all_preds == all_labels).mean()
        cm       = self._confusion_matrix(all_labels, all_preds)

        print(f"Test Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")
        self._per_class(all_labels, all_preds)
        return accuracy, cm

    def _confusion_matrix(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        matrix = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=int)
        for true, pred in zip(labels, preds):
            matrix[true][pred] += 1
        return matrix

    def _per_class(self, labels: np.ndarray, preds: np.ndarray):
        for i, cls in enumerate(config.IRIS_CLASSES):
            mask = labels == i
            if mask.sum() == 0:
                continue
            cls_acc = (preds[mask] == i).mean()
            print(f"  {cls:<22} {cls_acc:.4f}  ({int(cls_acc * mask.sum())}/{mask.sum()})")
