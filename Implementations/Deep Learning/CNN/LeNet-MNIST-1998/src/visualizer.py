import numpy as np
import matplotlib.pyplot as plt

import config


def plot_training_history(history: dict, model_name: str = "lenet5"):
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

    plt.suptitle(f"LeNet-5 Training History — MNIST", fontsize=13)
    plt.tight_layout()
    path = config.RESULTS_DIR / f"{model_name}_training_history.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()


def plot_confusion_matrix(matrix: np.ndarray, model_name: str = "lenet5"):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
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


def plot_sample_predictions(model, test_loader, model_name: str = "lenet5", num_samples: int = 10):
    import torch
    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    with torch.no_grad():
        preds = model(images.to(config.DEVICE)).argmax(1).cpu()

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(14, 5))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(images[i].squeeze(), cmap="gray")
        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"T:{labels[i].item()} P:{preds[i].item()}", color=color, fontsize=9)
        ax.axis("off")

    plt.suptitle("Sample Predictions (green=correct, red=wrong)")
    plt.tight_layout()
    path = config.RESULTS_DIR / f"{model_name}_sample_predictions.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()
