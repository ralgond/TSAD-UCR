import os

from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np
import random
import tsaug
import rolling

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



def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Remove randomness (may be slower on Tesla GPUs) 
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_window_list(data, win_size):
    ret = []

    for i in range(len(data)-win_size+1):
        term = data[i:i+win_size]
        ret.append(term)

    return ret

def minmax_scale(l_):
    l = np.array(l_)
    l_min = l.min()
    l_max = l.max()
    ret = (l - l_min) / (l_max - l_min)
    return ret.tolist()

def tail_padding_zero(l, length):
    ret = []
    ret.extend(l)
    if length - len(l) <= 0:
        return ret
    for i in range(length - len(l)):
        ret.append(0)
    return ret

def aggregate(ts, win_size=10):
    return list(rolling.Mean(ts, win_size))

def augament(ts):
    ret = []
    for _ in range(1,3):
        i = random.randint(0,100) % 9
        X = np.array(ts)
        if i == 0:
            ret.append(tsaug.AddNoise(scale=0.1).augment(X))
        if i == 1:
            ret.append(tsaug.Convolve(window="flattop", size=20).augment(X))
        if i == 2:
            term = tsaug.Crop(size=100).augment(X)
            ret.append(tail_padding_zero(term, len(X)))
        if i == 3:
            ret.append(tsaug.Drift(max_drift=0.7, n_drift_points=20).augment(X))
        if i == 4:
            ret.append(tsaug.Pool(size=40).augment(X))
        if i == 5:
            ret.append(tsaug.Quantize(n_levels=100).augment(X))
        if i == 6:
            term = tsaug.Resize(size=100).augment(X)
            ret.append(tail_padding_zero(term, len(X)))
        if i == 7:
            ret.append(tsaug.Reverse().augment(X))
        if i == 8:
            ret.append(tsaug.TimeWarp(n_speed_change=20, max_speed_ratio=6).augment(X))
    
    return ret


class DataWithLable:
    def __init__(self, slice, label) -> None:
        self.slice = slice
        self.label = label

class TrainDataset(Dataset):
    def __init__(self, pos_ts, neg_ts) -> None:
        super().__init__()
        self.ts = []
        for pos in pos_ts:
            self.ts.append(DataWithLable(pos, 1))
        for neg in neg_ts:
            self.ts.append(DataWithLable(neg, 0))

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        item = self.ts[index]
        return torch.tensor(item.slice, dtype=torch.float32), torch.tensor(item.label, dtype=torch.float32)

class TestDataset(Dataset):
    def __init__(self, ts) -> None:
        super().__init__()
        self.ts = ts

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        item = self.ts[index]
        return torch.tensor(item, dtype=torch.float32)

class residual_block:
    def __init__(self, desc) -> None:
        pass
    def __enter__ (self):
        pass
    def __exit__ (self, exc_type, exc_value, traceback):
        pass