from pathlib import Path

BASE_DIR = Path(__file__).parent

DATA_DIR      = BASE_DIR / "data" / "extracted" / "cifar-10-batches-py"
MODELS_DIR    = BASE_DIR / "models"
RESULTS_DIR   = BASE_DIR / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE    = 128
LEARNING_RATE = 0.001
EPOCHS        = 30
DEVICE        = "cuda"

NUM_CLASSES = 10
IMAGE_SIZE  = 32

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
