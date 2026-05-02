import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config

_CACHE = config.BASE_DIR / "data" / "iris_cache.npz"

_SEED  = 42
_TRAIN = 0.70
_VAL   = 0.15


def load_iris_raw() -> tuple[np.ndarray, np.ndarray]:
    """
    Fetch Iris from UCI ML Repository via ucimlrepo.
    Results are cached to data/iris_cache.npz after the first download
    so subsequent runs do not require network access.
    """
    if _CACHE.exists():
        data = np.load(_CACHE)
        return data["X"].astype(np.float32), data["y"].astype(np.int64)

    from ucimlrepo import fetch_ucirepo

    print("Fetching Iris dataset from UCI ML Repository...")
    iris  = fetch_ucirepo(id=53)
    X     = iris.data.features.values.astype(np.float32)
    y_raw = iris.data.targets.values.ravel()

    le = LabelEncoder()
    y  = le.fit_transform(y_raw).astype(np.int64)

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_CACHE, X=X, y=y)
    print(f"Cached at {_CACHE}")
    return X, y


def _split(X: np.ndarray, y: np.ndarray):
    rng     = np.random.RandomState(_SEED)
    idx     = rng.permutation(len(X))
    X, y    = X[idx], y[idx]
    n       = len(X)
    n_train = int(_TRAIN * n)
    n_val   = int(_VAL   * n)
    return (
        X[:n_train],            y[:n_train],
        X[n_train:n_train+n_val], y[n_train:n_train+n_val],
        X[n_train+n_val:],      y[n_train+n_val:],
    )


class IrisDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data() -> tuple[DataLoader, DataLoader, DataLoader]:
    X, y = load_iris_raw()
    X_train, y_train, X_val, y_val, X_test, y_test = _split(X, y)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    train_loader = DataLoader(IrisDataset(X_train_s, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(IrisDataset(X_val_s,   y_val),   batch_size=config.BATCH_SIZE)
    test_loader  = DataLoader(IrisDataset(X_test_s,  y_test),  batch_size=config.BATCH_SIZE)

    print(f"Data split — Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")
    return train_loader, val_loader, test_loader


def load_raw_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unscaled train/test arrays for classical ML baselines."""
    X, y = load_iris_raw()
    X_train, y_train, _, _, X_test, y_test = _split(X, y)
    return X_train, y_train, X_test, y_test
