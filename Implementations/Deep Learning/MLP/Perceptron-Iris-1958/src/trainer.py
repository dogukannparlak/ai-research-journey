import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config


class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
        self.model        = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = nn.CrossEntropyLoss()
        self.optimizer    = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def _run_epoch(self, loader: DataLoader, training: bool):
        self.model.train() if training else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(training):
            for X, y in loader:
                X, y    = X.to(config.DEVICE), y.to(config.DEVICE)
                outputs = self.model(X)
                loss    = self.criterion(outputs, y)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * y.size(0)
                correct    += (outputs.argmax(1) == y).sum().item()
                total      += y.size(0)

        return total_loss / total, correct / total

    def train(self, epochs: int = config.EPOCHS):
        print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
        print("-" * 55)
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(self.train_loader, training=True)
            val_loss,   val_acc   = self._run_epoch(self.val_loader,   training=False)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if epoch % 20 == 0 or epoch == 1:
                print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>10.4f} {val_loss:>10.4f} {val_acc:>9.4f}")

        return self.history

    def save(self, filename: str):
        path = config.MODELS_DIR / filename
        torch.save(self.model.state_dict(), path)
        print(f"Model saved: {path}")
