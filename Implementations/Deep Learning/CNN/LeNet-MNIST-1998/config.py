from pathlib import Path

BASE_DIR = Path(__file__).parent

DATA_PATH   = BASE_DIR / "data" / "mnist.npz"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE    = 64
LEARNING_RATE = 0.001
EPOCHS        = 10
DEVICE        = "cpu"

NUM_CLASSES = 10
IMAGE_SIZE  = 28
