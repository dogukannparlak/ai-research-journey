"""
Extract cifar-10-python.tar.gz into data/extracted/.

Run once before training:
    python setup_data.py
"""

import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config

RAW_ARCHIVE  = Path(__file__).parent / "data" / "raw" / "cifar-10-python.tar.gz"
EXTRACT_DIR  = Path(__file__).parent / "data" / "extracted"


def main() -> None:
    if not RAW_ARCHIVE.exists():
        print(f"Error: archive not found at {RAW_ARCHIVE}")
        print("Place cifar-10-python.tar.gz in data/raw/ and try again.")
        sys.exit(1)

    expected = config.DATA_DIR
    if expected.exists() and any(expected.iterdir()):
        print(f"Already extracted: {expected}")
        print("Delete data/extracted/ manually to re-extract.")
        return

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {RAW_ARCHIVE.name} ...")
    with tarfile.open(RAW_ARCHIVE, "r:gz") as tar:
        tar.extractall(EXTRACT_DIR)

    print(f"Done. Data available at: {expected}")

    batches = list(expected.glob("data_batch_*"))
    print(f"Found {len(batches)} training batches + test_batch")


if __name__ == "__main__":
    main()
