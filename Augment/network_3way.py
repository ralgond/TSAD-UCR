import torch
import torch.nn as nn
import torch.nn.functional as F

class Cnn3way(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(3)
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

        self.conv4 = nn.Conv1d(8, 16, 15, padding='same')
        self.bn4 = nn.BatchNorm1d(16)

        self.conv4_1 = nn.Conv1d(16, 32, 15, padding='same')
        self.bn4_1 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 1)

    def merge(self, x, y, z):
        out = torch.cat((x, y, z), dim=1)
        return out

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

        z = self.conv4(x0)
        z = self.bn4(z)
        z = F.relu(z)
        z = self.pool(z)

        z = self.conv4_1(z)
        z = self.bn4_1(z)
        z = F.relu(z)
        z = self.pool(z)

        #print ("x.shape:",x.shape,"y.shape:",y.shape)
        out = self.merge(x, y, z)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)

        return torch.sigmoid(out)


class Cnn3wayAdd(Cnn3way):
    def __init__(self) -> None:
        super().__init__()
    def merge(self, x, y, z):
        return torch.add(x, y, z)
