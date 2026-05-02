from pathlib import Path

BASE_DIR = Path(__file__).parent

MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE    = 16
LEARNING_RATE = 0.01
EPOCHS        = 200
DEVICE        = "cpu"

NUM_CLASSES = 3
INPUT_DIM   = 4
HIDDEN_DIM  = 16

IRIS_CLASSES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
