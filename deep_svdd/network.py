import torch.nn as nn
from common import Sigmoid, Interpolation

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        self.channel = channel
    def forward(self, x):
        return x.view(x.size(0), self.channel, -1)

class Print(nn.Module):
    def __init__(self, info) -> None:
        super().__init__()
        self.info = info
    def forward(self, x):
        print(self.info, x.shape)
        return x

class AE_Conv_128(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, 5, padding=2, bias=False),
            nn.BatchNorm1d(8, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool1d(2), # 64

            nn.Conv1d(8, 4, 5, padding=2, bias=False),
            nn.BatchNorm1d(4, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool1d(2), #32

            Flatten(),
            #Print("A"),
            nn.Linear(128, 32, bias=False), # compress into a (32,) vector
            #Print("B"),
        )
        self.encoder.__setattr__("rep_dim", 32)

        self.decoder = nn.Sequential(
            UnFlatten(4),
            #Print("C"),

            Interpolation(4),
            nn.ConvTranspose1d(4, 8, 5, padding=2, bias=False),
            nn.ReLU(),
            
            Interpolation(4),
            nn.ConvTranspose1d(8, 1, 5, padding=2, bias=False),
            
            Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

