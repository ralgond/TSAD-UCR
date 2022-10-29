from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CHANNEL_1 = 16
CHANNEL_2 = 32

CHANNEL_OUTPUT = 1
EPOCHS = 5
BATCH_SIZE = 256

KERNEL_SIZE = 15

L2_REG = 0.005
L1_REG = 0.02

class LukasEncoder(nn.Module):
    def __init__(self, win_size: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(4)
        self.win_size = win_size
        self.rep_dim = self.win_size//16
        
        # enconder
        self.conv1 = nn.Conv1d(1, CHANNEL_1, KERNEL_SIZE, bias=False, padding="same")
        self.bn1 = nn.BatchNorm1d(CHANNEL_1, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(CHANNEL_1, CHANNEL_2, 5, bias=False, padding="same")
        self.bn2 = nn.BatchNorm1d(CHANNEL_2, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(256, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.pool(x)

        #print ("block1.shape:", x.shape)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.pool(x)

        #print ("block2.shape:", x.shape)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
    
class LukasAE(nn.Module):
    def __init__(self, win_size: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(4)
        self.win_size = win_size        
        self.rep_dim = self.win_size//16

        # enconder
        self.conv1 = nn.Conv1d(1, CHANNEL_1, KERNEL_SIZE, bias=False, padding="same")
        self.bn1 = nn.BatchNorm1d(CHANNEL_1, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(CHANNEL_1, CHANNEL_2, 5, bias=False, padding="same")
        self.bn2 = nn.BatchNorm1d(CHANNEL_2, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(256, self.rep_dim, bias=False)

        # decoder
        self.conv3 = nn.Conv1d(1, CHANNEL_2, KERNEL_SIZE, bias=False, padding="same")
        self.bn3 = nn.BatchNorm1d(CHANNEL_2, eps=1e-04, affine=False)
        self.conv4 = nn.Conv1d(CHANNEL_2, CHANNEL_1, KERNEL_SIZE, bias=False, padding="same")
        self.bn4 = nn.BatchNorm1d(CHANNEL_1, eps=1e-04, affine=False)

        self.conv5 = nn.Conv1d(CHANNEL_1, CHANNEL_OUTPUT, KERNEL_SIZE, bias=False, padding="same")
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.pool(x)

        #print ("block1.shape:", x.shape)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.pool(x)

        #print ("block2.shape:", x.shape)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        x = x.view(x.size(0), 1, self.win_size//16)

        #print ("block3.shape:", x.shape)

        
        x = F.interpolate(x, scale_factor=4)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)

        x = F.interpolate(x, scale_factor=4)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x)

        x = self.conv5(x)
        x = torch.sigmoid(x)

        return x

if __name__ == "__main__":
    data = torch.randn(20, 1, 128)
    ae = LukasAE(None, 128)
    result = ae(data)
    print (result.shape)
