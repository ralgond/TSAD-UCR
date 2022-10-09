from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from ucr_dataset import get_data
from common import UCRDataset, UCRDatasetForTest, Sigmoid, Interpolation, set_seed
import time
import matplotlib.pyplot as plt

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
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

            Print("A"),
            nn.Linear(4 * 32, 32, bias=False)
        )

        self.decoder = nn.Sequential(
            Interpolation(2),
            nn.ConvTranspose1d(4, 8, 5, padding=2),
            nn.ReLU(),
            
            Interpolation(2),
            nn.ConvTranspose1d(8, 1, 5, padding=2),
            
            Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == "__main__":
    data = torch.randn(20, 1, 128)
    ae = AE_Conv_128()
    ae(data)