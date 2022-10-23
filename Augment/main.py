from cProfile import label
import random
from ucr_dataset import get_series
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import time
import tsaug

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from common import set_seed
import rolling

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
    for i in range(length - len(l)):
        ret.append(0)
    return ret


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

class AugmentNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(1, 8, 15, padding='same')
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, 15, padding='same')
        self.bn2 = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return torch.sigmoid(x)

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


class Channel:
    def __init__(self, id, pos_samples, neg_samples) -> None:
        self.id = id
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.model = AugmentNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 5e-4)
        self.loss_fn = torch.nn.BCELoss()

        self.model.cuda()

    def train(self):
        print (f"==========>pos_samples.len: {len(self.pos_samples)}, neg_samples.len: {len(self.neg_samples)}")

        dataset = TrainDataset(self.pos_samples, self.neg_samples)
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in range(10):
            loss_epoch = 0.
            batch_epoch = 0
            self.model.train()
            #print ("train_data.len:", len(dataset))
            for i, (slice, label) in enumerate(train_loader):
                self.optimizer.zero_grad()

                input_data = slice.cuda()
                input_data = input_data.unsqueeze(1) #增加通道
                out = self.model(input_data)
                out = out.squeeze(-1).float()

                label = torch.tensor(label).cuda()
                loss = self.loss_fn(out, label)

                loss.backward()
                self.optimizer.step()

                batch_epoch += 1
                loss_epoch += loss.cpu().item()
            print (f"epoch:{epoch}, loss:{loss_epoch/batch_epoch:.5f}")
            
    def predict(self, test_list):
        self.model.eval()
        
        with torch.no_grad():
            dataset = TestDataset(test_list)
            test_loader = DataLoader(dataset, batch_size=128, shuffle=False)
            print ("==============>test_list.len:", len(test_list)) 
            print ("==============>dataset:", len(dataset)) 
            scores = []
            for i, (slice) in enumerate(test_loader):
                input_data = slice.cuda()
                input_data = input_data.unsqueeze(1) #增加通道
                out = self.model(input_data)
                out = out.squeeze(-1).float()

                for item in out:
                    scores.append(item.cpu().item())
            print ("==============>scores.len:", len(scores))        
        return scores

def aggregate(ts, win_size=10):
    return list(rolling.Mean(ts, win_size))

def main(file_no):
    set_seed(file_no)

    all_data, split_pos, anomaly_range = get_series(file_no)

    all_data = minmax_scale(all_data)

    #all_data = aggregate(all_data, win_size=10)

    train_data, test_data = all_data[:split_pos], all_data[split_pos:]

    train_pos_list = create_window_list(train_data, 128)

    start_time = time.time()
    train_neg_data = []
    for train_pos in train_pos_list:
        augamented_neg_list = augament(train_pos)
        for augamented_neg in augamented_neg_list:
            train_neg_data.append(augamented_neg)
    print("aug_time:", (time.time() - start_time))

    channel = Channel(file_no, train_pos_list, train_neg_data)
    channel.train()

    test_list = create_window_list(test_data, 128)
    scores = channel.predict(test_list=test_list)

    correct_range = (anomaly_range[0]-100, anomaly_range[1]+100)
    pos = np.argmin(scores) + len(train_data)
    if pos >= correct_range[0] and pos <= correct_range[1]:
        return 1
    else:
        return -1

if __name__ == "__main__":
    correct_cnt = 0
    error_cnt = 0

    ret = None
    for i in range(1,251):
        if i in [239,240,241]:
            ret = -1
        else:
            ret = main(i)

        if (ret > 0):
            correct_cnt += 1
            print (f"({i}) correct, ==========>correct_cnt:{correct_cnt}, error_cnt:{error_cnt}")
        else:
            error_cnt += 1
            print (f"({i}) error, ============>correct_cnt:{correct_cnt}, error_cnt:{error_cnt}")