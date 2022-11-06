import torch
import torch.nn as nn
import torch.nn.functional as F

class Cnn2way(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool1d(2)
        self.conv1 = nn.Conv1d(1, 8, 31, padding='same')
        self.bn1 = nn.BatchNorm1d(8)

        self.conv2 = nn.Conv1d(8, 16, 7, padding='same')
        self.bn2 = nn.BatchNorm1d(16)

        self.conv2_1 = nn.Conv1d(16, 32, 7, padding='same')
        self.bn2_1 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(8, 16, 11, padding='same')
        self.bn3 = nn.BatchNorm1d(16)

        self.conv3_1 = nn.Conv1d(16, 32, 11, padding='same')
        self.bn3_1 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = F.relu(x0)
        x0 = self.pool(x0)

        x = self.conv2(x0)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = F.relu(x)
        x = self.pool(x)

        y = self.conv3(x0)
        y = self.bn3(y)
        y = F.relu(y)
        y = self.pool(y)

        y = self.conv3_1(y)
        y = self.bn3_1(y)
        y = F.relu(y)
        y = self.pool(y)

        #print ("x.shape:",x.shape,"y.shape:",y.shape)
        out = torch.cat((x, y), dim=1)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)

        return torch.sigmoid(out)


