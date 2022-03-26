import torch
import torchvision.models

from spp import SPPLayer
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, ReLU, Dropout, Softmax, BatchNorm2d, Flatten


class AudioDFNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = Sequential(
            Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            ReLU(),
            BatchNorm2d(32),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            ReLU(),
            BatchNorm2d(32),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            ReLU(),
            SPPLayer(3)
            # MaxPool2d(2),
        )
        self.classifier = Sequential(
            Flatten(),
            # Linear(1024, 64),
            Linear(896, 64),
            ReLU(),
            Dropout(0.2),
            Linear(64, 3)
            # Softmax(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
