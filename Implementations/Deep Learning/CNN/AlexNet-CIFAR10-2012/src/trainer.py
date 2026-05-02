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
        # Reduce LR by 0.1 when validation loss plateaus for 5 epochs
        self.scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def _run_epoch(self, loader: DataLoader, training: bool) -> tuple[float, float]:
        self.model.train() if training else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(training):
            for images, labels in loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * labels.size(0)
                correct    += (outputs.argmax(1) == labels).sum().item()
                total      += labels.size(0)

        return total_loss / total, correct / total

    def train(self, epochs: int = config.EPOCHS) -> dict:
        print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10}")
        print("-" * 65)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(self.train_loader, training=True)
            val_loss,   val_acc   = self._run_epoch(self.val_loader,   training=False)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>10.4f} {val_loss:>10.4f} {val_acc:>9.4f} {current_lr:>10.2e}")

        return self.history

    def save(self, filename: str = "alexnet.pt") -> None:
        path = config.MODELS_DIR / filename
        torch.save(self.model.state_dict(), path)
        print(f"Model saved: {path}")
