import torch.nn as nn


class Perceptron(nn.Module):
    """
    Rosenblatt (1958) — single-layer linear classifier.

    Multi-class extension of the original binary Perceptron:
    weight matrix W ∈ ℝ^{input_dim × num_classes}.

    Decision boundaries are strictly linear hyperplanes.
    The Perceptron convergence theorem guarantees convergence
    only for linearly separable data — a constraint that Iris
    partially violates (Versicolor and Virginica overlap).
    """

    def __init__(self, input_dim: int = 4, num_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    """
    Two-layer MLP — introduces one hidden layer with ReLU activation.

    A single hidden layer is sufficient (by the Universal Approximation
    Theorem, Cybenko 1989) to learn any continuous function, including
    the non-linear boundary between Versicolor and Virginica.

    Architecture: Linear(4 → 16) → ReLU → Linear(16 → 3)
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
