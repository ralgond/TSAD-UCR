import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from ucr_dataset import get_data
from common import UCRDataset, UCRDatasetForTest, Sigmoid, Interpolation, set_seed
import time
import matplotlib.pyplot as plt
from ae_trainer import train

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
            Interpolation(2),
            nn.ConvTranspose1d(4, 8, 5, padding=2, bias=False),
            nn.ReLU(),
            
            Interpolation(2),
            nn.ConvTranspose1d(8, 1, 5, padding=2, bias=False),
            
            Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == "__main__":
    set_seed(42)

    train_data, test_data = get_data(1)

    ae = AE_Conv()
    loader_for_train = DataLoader(UCRDataset(train_data, 128), batch_size=512, shuffle=False)
    optimizer = optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    ae = train(train_data, test_data, ae, loader_for_train, optimizer, criterion, add_channel=True)

    loader_for_test = DataLoader(UCRDatasetForTest(train_data, 128), batch_size=100)

    logits_data = []

    ae.eval()
    with torch.no_grad():
        for data in loader_for_test:
            data = data.unsqueeze(1)
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