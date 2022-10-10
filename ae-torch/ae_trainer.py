import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from ucr_dataset import get_data
from common import UCRDataset, UCRDatasetForTest, Sigmoid
import time
import matplotlib.pyplot as plt

def train(train_data, test_data, ae, loader_for_train, optimizer, criterion, add_channel=False, n_epoch=10):
        
    ae.to("cuda")

    ae.train()
    for epoch in range(n_epoch):
        loss = 0
        start_time = time.time()
        for data in loader_for_train:
            if add_channel:
                data = data.unsqueeze(1)

            data = data.to("cuda")

            optimizer.zero_grad()

            logits = ae(data)

            train_loss = criterion(logits, data)

            train_loss.backward()

            optimizer.step()

            loss += train_loss.item()

        end_time = time.time()

        loss = loss / len(loader_for_train)

        print ("epoch:{}, time: {:.2f}s, loss: {:.6f}".format(epoch, (end_time-start_time), loss))

    return ae