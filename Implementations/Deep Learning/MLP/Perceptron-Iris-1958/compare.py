"""
compare.py — Run after train.py.

Loads saved Perceptron and MLP weights, trains classical ML baselines
from scratch, and produces a full comparison table + bar chart.
"""
import sys
import csv
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from src.dataset    import load_data, load_raw_arrays
from src.model      import Perceptron, MLP
from src.evaluator  import Evaluator
from src.classical  import run_svm, run_knn, run_logistic
from src.visualizer import plot_confusion_matrix, plot_accuracy_comparison


def load_perceptron() -> Perceptron:
    model = Perceptron(config.INPUT_DIM, config.NUM_CLASSES)
    model.load_state_dict(torch.load(config.MODELS_DIR / "perceptron.pt", map_location=config.DEVICE))
    model.eval()
    return model


def load_mlp() -> MLP:
    model = MLP(config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_CLASSES)
    model.load_state_dict(torch.load(config.MODELS_DIR / "mlp.pt", map_location=config.DEVICE))
    model.eval()
    return model


def save_comparison(results: list[dict]):
    path   = config.RESULTS_DIR / "comparison.csv"
    fields = ["model", "test_accuracy", "train_time_s"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nComparison table saved: {path}")


def print_table(results: list[dict]):
    print("\n" + "=" * 55)
    print(f"  {'Model':<20} {'Accuracy':>10} {'Train Time':>12}")
    print("-" * 55)
    for r in results:
        t = f"{r['train_time_s']:.3f}s" if r["train_time_s"] is not None else "—"
        print(f"  {r['model']:<20} {r['test_accuracy']:>10.4f} {t:>12}")
    print("=" * 55)


def main():
    print("=" * 55)
    print("  Classical ML vs Perceptron vs MLP — Iris")
    print("=" * 55)

    _, _, test_loader            = load_data()
    X_train, y_train, X_test, y_test = load_raw_arrays()
    results = []

    print("\n[Perceptron] Loading saved model...")
    perc_acc, perc_cm = Evaluator(load_perceptron(), test_loader).evaluate()
    plot_confusion_matrix(perc_cm, model_name="perceptron")
    results.append({"model": "Perceptron", "test_accuracy": round(perc_acc, 4), "train_time_s": None})

    print("\n[MLP] Loading saved model...")
    mlp_acc, mlp_cm = Evaluator(load_mlp(), test_loader).evaluate()
    plot_confusion_matrix(mlp_cm, model_name="mlp")
    results.append({"model": "MLP", "test_accuracy": round(mlp_acc, 4), "train_time_s": None})

    svm_acc, svm_cm, svm_time = run_svm(X_train, y_train, X_test, y_test)
    plot_confusion_matrix(svm_cm, model_name="svm")
    results.append({"model": "SVM (RBF)", "test_accuracy": round(float(svm_acc), 4), "train_time_s": round(svm_time, 3)})

    knn_acc, knn_cm, knn_time = run_knn(X_train, y_train, X_test, y_test, k=5)
    plot_confusion_matrix(knn_cm, model_name="knn")
    results.append({"model": "kNN (k=5)", "test_accuracy": round(float(knn_acc), 4), "train_time_s": round(knn_time, 3)})

    lr_acc, lr_cm, lr_time = run_logistic(X_train, y_train, X_test, y_test)
    plot_confusion_matrix(lr_cm, model_name="logistic")
    results.append({"model": "Logistic Reg.", "test_accuracy": round(float(lr_acc), 4), "train_time_s": round(lr_time, 3)})

    print_table(results)
    save_comparison(results)
    plot_accuracy_comparison(results)


if __name__ == "__main__":
    main()
