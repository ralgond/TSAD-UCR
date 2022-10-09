from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

class UCRDataset(Dataset):
    def __init__(self, ts, win_size) -> None:
        super().__init__()
        self.ts = ts
        self.win_size = win_size
    
    def __len__(self):
        return len(self.ts) - self.win_size + 1

    def __getitem__(self, index):
        return torch.tensor(self.ts[index:index+self.win_size], dtype=torch.float32)

class UCRDatasetForTest(Dataset):
    def __init__(self, ts, win_size) -> None:
        super().__init__()
        self.ts = ts
        self.win_size = win_size
    
    def __len__(self):
        return len(self.ts) // self.win_size

    def __getitem__(self, index):
        return torch.tensor(self.ts[self.win_size*index : self.win_size*index+self.win_size], dtype=torch.float32)


class Sigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

class Interpolation(nn.Module):
    def __init__(self, scale_factor) -> None:
        super().__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)
