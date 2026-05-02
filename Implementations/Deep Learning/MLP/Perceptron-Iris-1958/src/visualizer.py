import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import config


def plot_training_history(history: dict, model_name: str = "model"):
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
    ax2.set_ylim(0, 1.05)
    ax2.legend()

    title = model_name.replace("_", " ").title()
    plt.suptitle(f"{title} Training History — Iris", fontsize=13)
    plt.tight_layout()
    path = config.RESULTS_DIR / f"{model_name}_training_history.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()


def plot_confusion_matrix(matrix: np.ndarray, model_name: str = "model"):
    short = [c.split("-")[1] for c in config.IRIS_CLASSES]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(short, rotation=15)
    ax.set_yticklabels(short)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name.upper()}")

    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, matrix[i, j], ha="center", va="center", color=color, fontsize=13)

    plt.tight_layout()
    path = config.RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()


def plot_pca_scatter(model: torch.nn.Module, test_loader: DataLoader, model_name: str = "model"):
    """
    Project the test set to 2D via PCA and compare true labels with model predictions.
    Misclassified points are highlighted with a red circle.
    """
    from sklearn.decomposition import PCA

    model.eval()
    all_X, all_y_true, all_y_pred = [], [], []

    with torch.no_grad():
        for X, y in test_loader:
            preds = model(X.to(config.DEVICE)).argmax(1).cpu().numpy()
            all_X.append(X.numpy())
            all_y_true.append(y.numpy())
            all_y_pred.append(preds)

    X_np   = np.concatenate(all_X)
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    pca  = PCA(n_components=2)
    X_2d = pca.fit_transform(X_np)

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    short  = [c.split("-")[1] for c in config.IRIS_CLASSES]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for cls_idx in range(config.NUM_CLASSES):
        mask = y_true == cls_idx
        ax1.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=colors[cls_idx], label=short[cls_idx],
                    alpha=0.85, edgecolors="k", linewidths=0.4, s=60)
    ax1.set_title("True Labels (PCA projection)")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.legend()

    for cls_idx in range(config.NUM_CLASSES):
        mask = y_pred == cls_idx
        ax2.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=colors[cls_idx], label=short[cls_idx],
                    alpha=0.85, edgecolors="k", linewidths=0.4, s=60)

    wrong = y_true != y_pred
    if wrong.any():
        ax2.scatter(X_2d[wrong, 0], X_2d[wrong, 1],
                    facecolors="none", edgecolors="red",
                    linewidths=2.0, s=130, label="misclassified")

    ax2.set_title(f"Predicted Labels — {model_name.upper()}")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.legend()

    plt.suptitle(f"PCA Projection — {model_name.upper()} on Iris Test Set", fontsize=13)
    plt.tight_layout()
    path = config.RESULTS_DIR / f"{model_name}_pca_scatter.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()


def plot_accuracy_comparison(results: list[dict]):
    names  = [r["model"] for r in results]
    accs   = [r["test_accuracy"] * 100 for r in results]
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, accs, color=colors[:len(names)], edgecolor="black", linewidth=0.7)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Model Comparison — Iris Dataset")
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    path = config.RESULTS_DIR / "comparison_bar.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.show()
