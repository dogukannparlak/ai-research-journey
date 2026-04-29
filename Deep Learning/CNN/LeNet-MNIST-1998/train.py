import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from src.dataset   import load_data
from src.model     import LeNet5, count_parameters
from src.trainer   import Trainer
from src.evaluator import Evaluator
from src.visualizer import (
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions,
)


def main():
    print("=" * 55)
    print("  LeNet-5 | MNIST | LeCun et al. 1998")
    print("=" * 55)

    train_loader, val_loader, test_loader = load_data()

    model = LeNet5(num_classes=config.NUM_CLASSES)
    print(f"\nModel: LeNet-5  |  Parameters: {count_parameters(model):,}\n")

    trainer = Trainer(model, train_loader, val_loader)
    history = trainer.train(epochs=config.EPOCHS)
    trainer.save("lenet5.pt")

    evaluator = Evaluator(model, test_loader)
    accuracy, confusion_matrix = evaluator.evaluate()

    plot_training_history(history)
    plot_confusion_matrix(confusion_matrix)
    plot_sample_predictions(model, test_loader)

    print("\nDone. Results saved to: results/")


if __name__ == "__main__":
    main()
