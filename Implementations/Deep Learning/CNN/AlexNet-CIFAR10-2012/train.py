import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from src.dataset    import load_data
from src.model      import AlexNet, count_parameters
from src.trainer    import Trainer
from src.evaluator  import Evaluator
from src.visualizer import (
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions,
)


def main() -> None:
    print("=" * 60)
    print("  AlexNet | CIFAR-10 | Krizhevsky et al. 2012")
    print("=" * 60)

    train_loader, val_loader, test_loader = load_data()

    model = AlexNet(num_classes=config.NUM_CLASSES)
    print(f"\nModel: AlexNet (CIFAR-10 adapted)  |  Parameters: {count_parameters(model):,}\n")

    trainer = Trainer(model, train_loader, val_loader)
    history = trainer.train(epochs=config.EPOCHS)
    trainer.save("alexnet.pt")

    evaluator = Evaluator(model, test_loader)
    accuracy, confusion_mat = evaluator.evaluate()

    plot_training_history(history)
    plot_confusion_matrix(confusion_mat)
    plot_sample_predictions(model, test_loader)

    print("\nDone. Results saved to: results/")


if __name__ == "__main__":
    main()
