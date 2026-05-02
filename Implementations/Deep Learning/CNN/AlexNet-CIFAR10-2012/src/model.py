import torch.nn as nn


class AlexNet(nn.Module):
    """
    Krizhevsky, Sutskever, and Hinton (2012) — ImageNet Classification with
    Deep Convolutional Neural Networks (NeurIPS 2012).

    Original AlexNet was designed for ImageNet (224×224, 1000 classes).
    This implementation adapts the architecture for CIFAR-10 (32×32, 10 classes):

      - Conv1 kernel reduced from 11×11 stride 4 → 3×3 stride 1 to preserve
        spatial resolution on small inputs.
      - MaxPool strides reduced to avoid collapsing 32×32 too early.
      - FC layers scaled down proportionally (4096 → 1024 → 512).
      - Local Response Normalization omitted (superseded by Batch Normalization).

    Key AlexNet innovations preserved:
      - ReLU activations (instead of sigmoid/tanh used in LeNet-5)
      - Dropout regularization in fully-connected layers
      - Overlapping max pooling (kernel 3, stride 2)
      - Deep stacked convolutional blocks (5 conv layers)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — 3×32×32 → 64×16×16
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Block 2 — 64×16×16 → 192×8×8
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Block 3 — 192×8×8 → 384×8×8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Block 4 — 384×8×8 → 256×8×8
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Block 5 — 256×8×8 → 256×4×4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
