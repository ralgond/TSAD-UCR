import torch
import torch.nn as nn
import torch.nn.functional as F

class Cnn2way(nn.Module):
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
        #=================================================
        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(64, 1)
        self.fc_dropout = nn.Dropout(0.5)
        #=================================================

        self.conv1_2 = nn.Conv1d(1, 8, 7, padding='same')
        self.bn1_2 = nn.BatchNorm1d(8)
        self.dropout1_2 = nn.Dropout(0.3)
        self.conv2_2 = nn.Conv1d(8, 16, 7, padding='same')
        self.bn2_2 = nn.BatchNorm1d(16)
        self.dropout2_2 = nn.Dropout(0.3)

        self.conv3_2 = nn.Conv1d(16, 16, 7, padding='same')
        self.bn3_2 = nn.BatchNorm1d(16)
        self.dropout3_2 = nn.Dropout(0.3)
        #==================================================

        # self.conv1_3 = nn.Conv1d(1, 8, 3, padding='same')
        # self.bn1_3 = nn.BatchNorm1d(8)
        # self.dropout1_3 = nn.Dropout(0.3)
        # self.conv2_3 = nn.Conv1d(8, 16, 3, padding='same')
        # self.bn2_3 = nn.BatchNorm1d(16)
        # self.dropout2_3 = nn.Dropout(0.3)

        # self.conv3_3 = nn.Conv1d(16, 16, 3, padding='same')
        # self.bn3_3 = nn.BatchNorm1d(16)
        # self.dropout3_3 = nn.Dropout(0.3)

    def forward(self, x):
        x0 = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        # =================================
        y = self.conv1_2(x0)
        y = self.bn1_2(y)
        y = F.relu(y)
        y = self.pool(y)

        y = self.conv2_2(y)
        y = self.bn2_2(y)
        y = F.relu(y)
        y = self.pool(y)

        y = self.conv3_2(y)
        y = self.bn3_2(y)
        y = F.relu(y)
        y = self.pool(y)
        # =================================
        # z = self.conv1_3(x0)
        # z = self.bn1_3(z)
        # z = F.relu(z)
        # z = self.pool(z)

        # z = self.conv2_3(z)
        # z = self.bn2_3(z)
        # z = F.relu(z)
        # z = self.pool(z)

        # z = self.conv3_3(z)
        # z = self.bn3_3(z)
        # z = F.relu(z)
        # z = self.pool(z)

        #xy = torch.add(x, y)
        xy = torch.div(torch.add(x, y), 2)

        out = torch.cat((x,y,xy), dim=1)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc_dropout(out)
        out = F.relu(out)
        out = self.fc2(out)

        return torch.sigmoid(out)


