import time
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def _prepare(X_train: np.ndarray, X_test: np.ndarray):
    """Apply StandardScaler — required for SVM and kNN distance metrics."""
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s


def run_svm(X_train, y_train, X_test, y_test):
    """
    RBF-kernel SVM — non-linear classifier that can separate
    Versicolor and Virginica in feature space via the kernel trick.
    """
    print("\n[SVM] Training RBF-SVC on Iris...")
    X_tr, X_te = _prepare(X_train, X_test)

    t0    = time.time()
    model = SVC(kernel="rbf", C=1.0, gamma="scale")
    model.fit(X_tr, y_train)
    train_time = time.time() - t0

    preds    = model.predict(X_te)
    accuracy = accuracy_score(y_test, preds)
    cm       = confusion_matrix(y_test, preds)

    print(f"[SVM] Test Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)  |  Train time: {train_time:.3f}s")
    return accuracy, cm, train_time


def run_knn(X_train, y_train, X_test, y_test, k: int = 5):
    """k-Nearest Neighbours — non-parametric, no training phase."""
    print(f"\n[kNN] Training k={k} on Iris...")
    X_tr, X_te = _prepare(X_train, X_test)

    t0    = time.time()
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_tr, y_train)
    train_time = time.time() - t0

    t1         = time.time()
    preds      = model.predict(X_te)
    infer_time = time.time() - t1

    accuracy = accuracy_score(y_test, preds)
    cm       = confusion_matrix(y_test, preds)

    print(f"[kNN] Test Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)  |  Train: {train_time:.3f}s  Inference: {infer_time:.3f}s")
    return accuracy, cm, train_time


def run_logistic(X_train, y_train, X_test, y_test):
    """
    Logistic Regression — the 'soft' statistical cousin of the Perceptron.
    Uses log-loss instead of the Perceptron criterion; learns probabilities
    rather than hard boundaries. Still linear, but regularised.
    """
    print("\n[Logistic Regression] Training on Iris...")
    X_tr, X_te = _prepare(X_train, X_test)

    t0    = time.time()
    model = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    model.fit(X_tr, y_train)
    train_time = time.time() - t0

    preds    = model.predict(X_te)
    accuracy = accuracy_score(y_test, preds)
    cm       = confusion_matrix(y_test, preds)

    print(f"[LR] Test Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)  |  Train time: {train_time:.3f}s")
    return accuracy, cm, train_time
