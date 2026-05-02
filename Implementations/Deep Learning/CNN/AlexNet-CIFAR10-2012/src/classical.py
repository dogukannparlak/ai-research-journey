import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def _prepare(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (N, 3, 32, 32) arrays and apply StandardScaler."""
    x_train_flat = x_train.reshape(len(x_train), -1).astype(np.float32) / 255.0
    x_test_flat  = x_test.reshape(len(x_test),  -1).astype(np.float32) / 255.0
    scaler       = StandardScaler()
    x_train_flat = scaler.fit_transform(x_train_flat)
    x_test_flat  = scaler.transform(x_test_flat)
    return x_train_flat, x_test_flat


def run_svm(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test:  np.ndarray, y_test:  np.ndarray,
) -> tuple[float, np.ndarray, float]:
    print("\n[SVM] Training LinearSVC on 50,000 samples...")
    x_tr, x_te = _prepare(x_train, x_test)

    t0    = time.time()
    model = LinearSVC(max_iter=3000, C=0.1)
    model.fit(x_tr, y_train)
    train_time = time.time() - t0

    preds    = model.predict(x_te)
    accuracy = accuracy_score(y_test, preds)
    cm       = confusion_matrix(y_test, preds)

    print(f"[SVM] Test Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)  |  Train time: {train_time:.1f}s")
    return accuracy, cm, train_time


def run_knn(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test:  np.ndarray, y_test:  np.ndarray,
    k: int = 5,
) -> tuple[float, np.ndarray, float]:
    print(f"\n[kNN] Training k={k} on 50,000 samples...")
    x_tr, x_te = _prepare(x_train, x_test)

    t0    = time.time()
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(x_tr, y_train)
    train_time = time.time() - t0

    t1         = time.time()
    preds      = model.predict(x_te)
    infer_time = time.time() - t1

    accuracy = accuracy_score(y_test, preds)
    cm       = confusion_matrix(y_test, preds)

    print(f"[kNN] Test Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)  |  Train: {train_time:.1f}s  Inference: {infer_time:.1f}s")
    return accuracy, cm, train_time
