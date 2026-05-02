import numpy as np
import matplotlib.pyplot as plt
import torch

import config


def plot_training_history(history: dict, model_name: str = "alexnet") -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"],   label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.suptitle("AlexNet Training History — CIFAR-10", fontsize=13)
    plt.tight_layout()
    path = config.RESULTS_DIR / f"{model_name}_training_history.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()


def plot_confusion_matrix(matrix: np.ndarray, model_name: str = "alexnet") -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.CIFAR10_CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(config.CIFAR10_CLASSES, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name.upper()}")

    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, matrix[i, j], ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    path = config.RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()


def plot_sample_predictions(model, test_loader, model_name: str = "alexnet", num_samples: int = 10) -> None:
    """Show sample predictions with true and predicted class names."""
    _MEAN = np.array([0.4914, 0.4822, 0.4465])
    _STD  = np.array([0.2470, 0.2435, 0.2616])

    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    with torch.no_grad():
        preds = model(images.to(config.DEVICE)).argmax(1).cpu()

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(16, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Denormalize for display
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * _STD + _MEAN).clip(0, 1)

        true_cls = config.CIFAR10_CLASSES[labels[i].item()]
        pred_cls = config.CIFAR10_CLASSES[preds[i].item()]
        color    = "green" if preds[i] == labels[i] else "red"

        ax.imshow(img)
        ax.set_title(f"T: {true_cls}\nP: {pred_cls}", color=color, fontsize=8)
        ax.axis("off")

    plt.suptitle("Sample Predictions (green=correct, red=wrong)")
    plt.tight_layout()
    path = config.RESULTS_DIR / f"{model_name}_sample_predictions.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()
