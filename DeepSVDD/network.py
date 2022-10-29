import torch
import torch.nn as nn
import torch.nn.functional as F


class UCRNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool1d(2)

        self.conv1 = nn.Conv1d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm1d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm1d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class UCRAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool1d(2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv1d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm1d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv1d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm1d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose1d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm1d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose1d(4, 8, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm1d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose1d(8, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(x.size(0), int(self.rep_dim / 16), -1)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)

        return x

if __name__ == "__main__":
    data = torch.randn(20, 1, 128)
    ae = UCRAutoencoder()
    result = ae(data)
    print("result.shape:", result.shape)