from ucr_dataset import get_series
import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from common import set_seed, SemiTrainDataset, TestDataset, minmax_scale, create_window_list, augament
from common import  UCRDataset, UCRDatasetForTest
from network import UCR_Network_Autoencoder, UCR_Network

if __name__ == "__main__":
    set_seed(42)

    WIN_SIZE = 128

    all_data, split_pos, anomaly_range = get_series(4)

    scaled_all_data = minmax_scale(all_data)

    train_data, test_data = scaled_all_data[:split_pos], scaled_all_data[split_pos:]

    train_dataset = UCRDataset(train_data, 128)

    scaled_all_dataset = UCRDatasetForTest(scaled_all_data, 128)


    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

    ae_net = UCR_Network_Autoencoder()
    ae_net = ae_net.cuda()

    criterion = nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(ae_net.parameters(), lr=0.0005)

    ae_net.train()
    for epoch in range(10):
        epoch_loss = 0.0
        n_batches = 0
        for batched_data in train_loader:

            optimizer.zero_grad()

            inputs = batched_data.cuda()

            #print("inputs.shape:", inputs.shape)
            outputs = ae_net(inputs)
            outputs = outputs.squeeze(1)
            #print("outputs.shape:", outputs.shape)

            outputs_loss = criterion(outputs, inputs)

            loss = torch.mean(outputs_loss)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print (f"Epoch:{epoch}, Train Loss:{epoch_loss / n_batches:.6f}")



    plt.figure(figsize=(12, 8))

    plt.plot([i for i in range(len(scaled_all_data))], scaled_all_data)

    ae_net.eval()
    predict_data = []

    with torch.no_grad():
        for slice in scaled_all_dataset:
            if len(slice) != 128:
                break
            outputs = ae_net(slice.cuda())
            #print ("outputs.shape:", outputs.shape)
            predict_data.extend(outputs.squeeze().cpu())

    print(len(predict_data))
    plt.plot([i for i in range(len(predict_data))], predict_data)

    plt.show()


