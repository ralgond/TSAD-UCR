from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleResnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(1, 8, 15, padding='same')
        self.bn1 = nn.BatchNorm1d(8)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(8, 16, 15, padding='same')
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(16, 16, 15, padding='same')
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv1d(16, 16, 15, padding='same')
        self.bn4 = nn.BatchNorm1d(16)
        self.dropout4 = nn.Dropout(0.3)

        self.fc0 = nn.Linear(2048, 256)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return torch.sigmoid(x)