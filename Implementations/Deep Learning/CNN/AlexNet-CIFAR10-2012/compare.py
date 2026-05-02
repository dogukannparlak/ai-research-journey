import sys
import csv
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from src.classical  import run_svm, run_knn
from src.dataset    import load_data, load_raw_arrays
from src.model      import AlexNet
from src.evaluator  import Evaluator
from src.visualizer import plot_confusion_matrix


def alexnet_accuracy() -> tuple[float, object]:
    """Load saved AlexNet weights and evaluate on test set."""
    _, _, test_loader = load_data()
    model = AlexNet(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(config.MODELS_DIR / "alexnet.pt", map_location=config.DEVICE))
    model.eval()

    evaluator = Evaluator(model, test_loader)
    return evaluator.evaluate()


def save_comparison(results: list[dict]) -> None:
    path   = config.RESULTS_DIR / "comparison.csv"
    fields = ["model", "test_accuracy", "train_time_s"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nComparison table saved: {path}")


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 55)
    print(f"{'Model':<14} {'Accuracy':>10} {'Train Time':>12}")
    print("-" * 55)
    for r in results:
        t = f"{r['train_time_s']:.1f}s" if r["train_time_s"] else "—"
        print(f"{r['model']:<14} {r['test_accuracy']:>10.4f} {t:>12}")
    print("=" * 55)


def main() -> None:
    print("=" * 60)
    print("  Classical ML vs AlexNet — CIFAR-10 Comparison")
    print("=" * 60)

    x_train, y_train, x_test, y_test = load_raw_arrays()
    results = []

    print("\n[AlexNet] Loading saved model...")
    alexnet_acc, alexnet_cm = alexnet_accuracy()
    plot_confusion_matrix(alexnet_cm, model_name="alexnet")
    results.append({
        "model":          "AlexNet",
        "test_accuracy":  round(float(alexnet_acc), 4),
        "train_time_s":   None,
    })

    svm_acc, svm_cm, svm_time = run_svm(x_train, y_train, x_test, y_test)
    plot_confusion_matrix(svm_cm, model_name="svm")
    results.append({
        "model":         "SVM",
        "test_accuracy": round(float(svm_acc), 4),
        "train_time_s":  round(svm_time, 1),
    })

    knn_acc, knn_cm, knn_time = run_knn(x_train, y_train, x_test, y_test, k=5)
    plot_confusion_matrix(knn_cm, model_name="knn")
    results.append({
        "model":         "kNN (k=5)",
        "test_accuracy": round(float(knn_acc), 4),
        "train_time_s":  round(knn_time, 1),
    })

    print_table(results)
    save_comparison(results)


if __name__ == "__main__":
    main()
