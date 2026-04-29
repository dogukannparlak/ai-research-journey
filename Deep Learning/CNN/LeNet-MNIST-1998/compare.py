import sys
import csv
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from src.classical  import run_svm, run_knn
from src.visualizer import plot_confusion_matrix


def load_raw():
    data = np.load(config.DATA_PATH)
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def lenet5_accuracy():
    """Read LeNet-5 result from its saved confusion matrix indirectly via model."""
    from src.dataset import load_data
    from src.model   import LeNet5
    from src.evaluator import Evaluator

    _, _, test_loader = load_data()
    model = LeNet5(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(config.MODELS_DIR / "lenet5.pt", map_location=config.DEVICE))
    model.eval()

    evaluator = Evaluator(model, test_loader)
    accuracy, cm = evaluator.evaluate()
    return accuracy, cm


def save_comparison(results: list[dict]):
    path = config.RESULTS_DIR / "comparison.csv"
    fields = ["model", "test_accuracy", "train_time_s"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nComparison table saved: {path}")


def print_table(results: list[dict]):
    print("\n" + "=" * 50)
    print(f"{'Model':<12} {'Accuracy':>10} {'Train Time':>12}")
    print("-" * 50)
    for r in results:
        t = f"{r['train_time_s']:.1f}s" if r["train_time_s"] else "—"
        print(f"{r['model']:<12} {r['test_accuracy']:>10.4f} {t:>12}")
    print("=" * 50)


def main():
    print("=" * 55)
    print("  Classical ML vs LeNet-5 — MNIST Comparison")
    print("=" * 55)

    x_train, y_train, x_test, y_test = load_raw()
    results = []

    print("\n[LeNet-5] Loading saved model...")
    lenet_acc, lenet_cm = lenet5_accuracy()
    plot_confusion_matrix(lenet_cm, model_name="lenet5")
    results.append({"model": "LeNet-5", "test_accuracy": round(float(lenet_acc), 4), "train_time_s": None})

    svm_acc, svm_cm, svm_time = run_svm(x_train, y_train, x_test, y_test)
    plot_confusion_matrix(svm_cm, model_name="svm")
    results.append({"model": "SVM", "test_accuracy": round(float(svm_acc), 4), "train_time_s": round(svm_time, 1)})

    knn_acc, knn_cm, knn_time = run_knn(x_train, y_train, x_test, y_test, k=3)
    plot_confusion_matrix(knn_cm, model_name="knn")
    results.append({"model": "kNN (k=3)", "test_accuracy": round(float(knn_acc), 4), "train_time_s": round(knn_time, 1)})

    print_table(results)
    save_comparison(results)


if __name__ == "__main__":
    main()
