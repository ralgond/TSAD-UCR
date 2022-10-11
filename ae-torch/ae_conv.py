import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from ucr_dataset import get_data, get_series
from common import UCRDataset, UCRDatasetForTest, SigmoidLayer, InterpolationLayer, set_seed
import time
import matplotlib.pyplot as plt
from ae_trainer import train
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class AE_Conv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, 5, padding=2, bias=False),
            nn.BatchNorm1d(8, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(8, 4, 5, padding=2, bias=False),
            nn.BatchNorm1d(4, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.decoder = nn.Sequential(
            InterpolationLayer(2),
            nn.ConvTranspose1d(4, 8, 5, padding=2, bias=False),
            nn.ReLU(),
            
            InterpolationLayer(2),
            nn.ConvTranspose1d(8, 1, 5, padding=2, bias=False),
            
            SigmoidLayer()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

WIN_SIZE = 128

def main(file_no:int):
    set_seed(42)

    all_data, split_pos, anomaly_range = get_series(file_no)
    train_data, test_data = get_data(file_no)

    ae = AE_Conv()
    loader_for_train = DataLoader(UCRDataset(train_data, WIN_SIZE), batch_size=512, shuffle=False)
    optimizer = optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    ae = train(train_data, test_data, ae, loader_for_train, optimizer, criterion, add_channel=True, n_epoch=30)

    loader_for_test = DataLoader(UCRDataset(test_data, WIN_SIZE), batch_size=512, shuffle=False)

    X_test = []
    for test in loader_for_test:
        for i in range(len(test)):
            X_test.append(test[i])
    
    # print ("==============>X_test.len:", len(X_test))
    # print ("==============>X_test[0].shape:", X_test[0].shape)

    X_pred = []
    ae.eval()
    with torch.no_grad():
        for data in loader_for_test:
            data = data.unsqueeze(1)
            data = data.to("cuda")
            logits = ae(data)
            logits = logits.cpu()
            for i in range(len(logits)):
                X_pred.append(logits[i].squeeze(0))

    # print ("==============>X_pred[-1].shape:", X_pred[-1].shape)
    
    X_score = []
    for test, pred in zip(X_test, X_pred):
        #print(np.array(test).shape, np.array(pred).shape)
        #score = euclidean_distances(np.expand_dims(test, 0), np.expand_dims(pred, 0))
        score = mean_squared_error(test, pred)
        X_score.append(score)

    # print ("+++++++++++++X_score.len:", len(X_score))
    # print ("+++++++++++++X_score[0].shape:", X_score[0].shape)
    correct_range = (anomaly_range[0]-100, anomaly_range[1]+100)
    pos = np.argmax(X_score) + len(train_data)
    if pos >= correct_range[0] and pos <= correct_range[1]:
        return 1
    else:
        return -1

if __name__ == "__main__":
    correct_count = 0
    error_count = 0
    for i in range(1,1+25):
        result = main(i)
        status = None
        if result > 0:
            correct_count += 1
            status = "correct"
        else:
            error_count += 1
            status = "error"
        print (f"({i}) {status}=========> correct:{correct_count}, error:{error_count}")






    # loader_for_test = DataLoader(UCRDatasetForTest(train_data, 128), batch_size=100)
    # logits_data = []
    # ae.eval()
    # with torch.no_grad():
    #     for data in loader_for_test:
    #         data = data.unsqueeze(1)
    #         data = data.to("cuda")
    #         logits = ae(data)
    #         logits_data.extend(logits.cpu().view(-1))

        
    # print("===========>len(train_data):", len(train_data))
    # print("===========>len(logits_data):", len(logits_data))
    # plt.figure(figsize=(16,9))
    # plt.plot([i for i in range(len(train_data))], train_data)
    # plt.plot([i for i in range(len(logits_data))], logits_data)
    # #print (logits_data[:100])
    # plt.show()