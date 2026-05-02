import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeCun et al. (1998) — Gradient-Based Learning Applied to Document Recognition
    Orijinal mimari: C1 → S2 → C3 → S4 → C5 → F6 → Output
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # C1: 1×28×28 → 6×24×24
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            # S2: 6×24×24 → 6×12×12
            nn.AvgPool2d(kernel_size=2, stride=2),

            # C3: 6×12×12 → 16×8×8
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            # S4: 16×8×8 → 16×4×4
            nn.AvgPool2d(kernel_size=2, stride=2),

            # C5: 16×4×4 → 120×1×1
            nn.Conv2d(16, 120, kernel_size=4),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            # F6
            nn.Linear(120, 84),
            nn.Tanh(),
            # Output
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
