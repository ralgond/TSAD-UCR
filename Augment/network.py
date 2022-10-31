import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCnnNet(nn.Module):
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

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
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

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return torch.sigmoid(x)

class UNet2(nn.Module):

    def ConvBNRelu(self, inc, outc, ks):
        return nn.Sequential(
            nn.Conv1d(inc, outc, ks, padding='same', bias=False),
            nn.BatchNorm1d(outc),
            nn.ReLU(),
            nn.Conv1d(outc, outc, ks, padding='same', bias=False),
            nn.BatchNorm1d(outc),
            nn.ReLU()
        )

    def UpsampleConv(self, inc, outc, ks):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(inc, outc, ks, padding='same'),
        )

    def LastLayer(self, inc, outc, ks):
        return nn.Sequential(
            nn.Conv1d(inc, outc, ks, padding='same')
        )

    def forward(self, x):
        conv1 = self.layer1(x)
        #print("conv1.shape:", conv1.shape)

        pool1 = self.pool(conv1)

        conv2 = self.layer2(pool1)
        #print("conv2.shape:", conv2.shape)

        pool2 = self.pool(conv2)

        conv3 = self.layer3(pool2)
        #print("conv3.shape:", conv3.shape)

        pool3 = self.pool(conv3)

        conv4 = self.layer4(pool3)
        #print("conv4.shape:", conv4.shape)

        pool4 = self.pool(conv4)

        conv5 = self.layer5(pool4)
        #print("conv5.shape:", conv5.shape)

        # upsample
        up6 = self.uc6(conv5)
        # print("conv4.shape:", conv4.shape)
        # print("up6.shape:", up6.shape)
        merge6 = torch.cat((conv4, up6), dim=1)
        # print("merge6.shape:", merge6.shape)
        conv6 = self.layer6(merge6)
        #print("conv6.shape:", conv6.shape)

        up7 = self.uc7(conv6)
        merge7 = torch.cat((conv3, up7), dim=1)
        conv7 = self.layer7(merge7)
        #print("conv7.shape:", conv7.shape)

        up8 = self.uc8(conv7)
        merge8 = torch.cat((conv2, up8), dim=1)
        conv8 = self.layer8(merge8)
        #print("conv8.shape:", conv8.shape)

        up9 = self.uc9(conv8)
        merge9 = torch.cat((conv1, up9), dim=1)
        conv9 = self.layer9(merge9)
        #print("conv9.shape:", conv9.shape)
        
        x = torch.flatten(self.output(conv9), 1)
        x = self.fc(x)
        return torch.sigmoid(x)

    def __init__(self) -> None:
        super().__init__()

        self.pool = nn.MaxPool1d(2)

        # 128
        self.layer1 = self.ConvBNRelu(1, 16, 3)
        # 64
        self.layer2 = self.ConvBNRelu(16, 32, 3)
        # 32
        self.layer3 = self.ConvBNRelu(32, 64, 3)
        # 16
        self.layer4 = self.ConvBNRelu(64, 128, 3)
        # 8
        self.layer5 = self.ConvBNRelu(128, 256, 3)
        # 16
        self.layer6 = self.ConvBNRelu(256, 128, 3)
        # 32
        self.layer7 = self.ConvBNRelu(128, 64, 3)
        # 64
        self.layer8 = self.ConvBNRelu(64, 32, 3)
        # 128
        self.layer9 = self.ConvBNRelu(32, 16, 3)


        self.uc6 = self.UpsampleConv(256, 128, 3)

        self.uc7 = self.UpsampleConv(128, 64, 3)

        self.uc8 = self.UpsampleConv(64, 32, 3)

        self.uc9 = self.UpsampleConv(32, 16, 3)

        self.output = self.LastLayer(16, 1, 3)

        self.fc = nn.Linear(128, 1)

class LSTM0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        #print ("out.shape:", out.shape)
        return torch.sigmoid(self.fc(out[:,-1,:]))


class UNet4(nn.Module):

    def ConvBNRelu(self, inc, outc, ks):
        return nn.Sequential(
            nn.Conv1d(inc, outc, ks, padding='same', bias=False),
            nn.BatchNorm1d(outc),
            nn.ReLU(),
            nn.Conv1d(outc, outc, ks, padding='same', bias=False),
            nn.BatchNorm1d(outc),
            nn.ReLU()
        )

    def UpsampleConv(self, inc, outc, ks):
        return nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv1d(inc, outc, ks, padding='same'),
        )

    def LastLayer(self, inc, outc, ks):
        return nn.Sequential(
            nn.Conv1d(inc, outc, ks, padding='same')
        )

    def forward(self, x):
        # 128
        conv1 = self.layer1(x)
        pool1 = self.pool(conv1)

        # 32
        conv2 = self.layer2(pool1)
        pool2 = self.pool(conv2)

        # 8
        conv3 = self.layer3(pool2)

        up3 = self.uc3(conv3)

        merge4 = torch.cat((conv2, up3), dim=1)
        # 32
        conv4 = self.layer4(merge4)

        up4 = self.uc4(conv4)
        merge5 = torch.cat((conv1, up4), dim=1)
        # 128
        conv5 = self.layer5(merge5)


        x = torch.flatten(self.output(conv5), 1)
        x = self.fc(x)
        return torch.sigmoid(x)

    def __init__(self) -> None:
        super().__init__()

        self.pool = nn.MaxPool1d(4)

        KS = 5

        # 128
        self.layer1 = self.ConvBNRelu(1, 16, KS)
        # 32
        self.layer2 = self.ConvBNRelu(16, 32, KS)
        # 8
        self.layer3 = self.ConvBNRelu(32, 64, KS)
        # 32
        self.layer4 = self.ConvBNRelu(64, 32, KS)
        # 128
        self.layer5 = self.ConvBNRelu(32, 16, KS)


        self.uc3 = self.UpsampleConv(64, 32, KS)

        self.uc4 = self.UpsampleConv(32, 16, KS)

        self.output = self.LastLayer(16, 1, KS)

        self.fc = nn.Linear(128, 1)


if __name__ == "__main__":

    net = UNet4()
    data = torch.randn(64, 1, 128)
    result = net(data)
    print(result.shape)

    # l = LSTM0(input_size=1, hidden_size=128, num_layers=2, num_classes=1)
    # data = torch.randn(64, 128, 1)
    # result = l(data)
    # print(result.shape)