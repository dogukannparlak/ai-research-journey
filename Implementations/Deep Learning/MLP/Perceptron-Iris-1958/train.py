import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from src.dataset    import load_data
from src.model      import Perceptron, MLP, count_parameters
from src.trainer    import Trainer
from src.evaluator  import Evaluator
from src.visualizer import (
    plot_training_history,
    plot_confusion_matrix,
    plot_pca_scatter,
)


def train_model(model, model_name, train_loader, val_loader, test_loader):
    slug = model_name.lower().replace(" ", "_")

    print(f"\n{'='*55}")
    print(f"  {model_name}  |  Parameters: {count_parameters(model):,}")
    print(f"{'='*55}")

    trainer = Trainer(model, train_loader, val_loader)
    history = trainer.train(epochs=config.EPOCHS)
    trainer.save(f"{slug}.pt")

    print(f"\n--- {model_name} Test Evaluation ---")
    evaluator = Evaluator(model, test_loader)
    accuracy, cm = evaluator.evaluate()

    plot_training_history(history, model_name=slug)
    plot_confusion_matrix(cm, model_name=slug)
    plot_pca_scatter(model, test_loader, model_name=slug)

    return float(accuracy)


def main():
    print("=" * 55)
    print("  Perceptron & MLP | Iris | Rosenblatt (1958)")
    print("=" * 55)

    train_loader, val_loader, test_loader = load_data()

    perc_acc = train_model(
        Perceptron(config.INPUT_DIM, config.NUM_CLASSES),
        "Perceptron",
        train_loader, val_loader, test_loader,
    )

    mlp_acc = train_model(
        MLP(config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_CLASSES),
        "MLP",
        train_loader, val_loader, test_loader,
    )

    print("\n" + "=" * 45)
    print(f"  {'Model':<15} {'Test Accuracy':>12}")
    print("-" * 45)
    print(f"  {'Perceptron':<15} {perc_acc:>12.4f}  ({perc_acc*100:.2f}%)")
    print(f"  {'MLP':<15} {mlp_acc:>12.4f}  ({mlp_acc*100:.2f}%)")
    print("=" * 45)
    print("\nDone. Results saved to: results/")


if __name__ == "__main__":
    main()
