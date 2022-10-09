import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from ucr_dataset import get_data
from common import UCRDataset, UCRDatasetForTest, Sigmoid
import time
import matplotlib.pyplot as plt
from ae_trainer import train

class AE_Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



if __name__ == "__main__":
    train_data, test_data = get_data(1)

    ae = AE_Linear()
    loader_for_train = DataLoader(UCRDataset(train_data, 128), batch_size=512, shuffle=False)
    optimizer = optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    #criterion = nn.BCELoss()

    ae = train(train_data, test_data, ae, loader_for_train, optimizer, criterion)

    loader_for_test = DataLoader(UCRDatasetForTest(train_data, 128), batch_size=100)

    logits_data = []

    ae.eval()
    with torch.no_grad():
        for data in loader_for_test:
            data = data.to("cuda")
            logits = ae(data)
            logits_data.extend(logits.cpu().view(-1))

        
    print("===========>len(train_data):", len(train_data))
    print("===========>len(logits_data):", len(logits_data))
    plt.figure(figsize=(16,9))
    plt.plot([i for i in range(len(train_data))], train_data)
    plt.plot([i for i in range(len(logits_data))], logits_data)
    #print (logits_data[:100])
    plt.show()
