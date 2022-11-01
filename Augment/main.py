from ucr_dataset import get_series
import numpy as np
import time


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from common import set_seed, TrainDataset, TestDataset, minmax_scale, create_window_list, augament
from network import SimpleCnnNet
from network_resnet import SimpleResnet
from network_multi_scale import MSResNet
from network_resnet import SimpleResnet

class Channel:
    def __init__(self, id, pos_samples, neg_samples) -> None:
        self.id = id
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.model = SimpleCnnNet()
        #self.model = UNet4()
        #self.model = SimpleResnet()
        #self.model = MSResNet(input_channel=1)
        #self.model = SimpleResnet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 5e-4)
        self.loss_fn = torch.nn.BCELoss()

        self.model.cuda()

    def train(self):
        print (f"==========>pos_samples.len: {len(self.pos_samples)}, neg_samples.len: {len(self.neg_samples)}")

        dataset = TrainDataset(self.pos_samples, self.neg_samples)
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        self.model.train()

        for epoch in range(15):
            loss_epoch = 0.
            batch_epoch = 0
            #print ("train_data.len:", len(dataset))
            for i, (slice, label) in enumerate(train_loader):
                self.optimizer.zero_grad()

                input_data = slice.cuda()
                input_data = input_data.unsqueeze(1) #增加通道
                out = self.model(input_data)
                out = out.squeeze(-1).float()

                label = label.cuda()
                loss = self.loss_fn(out, label)

                loss.backward()
                self.optimizer.step()

                batch_epoch += 1
                loss_epoch += loss.item()
            print (f"epoch:{epoch}, loss:{loss_epoch/batch_epoch:.5f}")
            
    def predict(self, test_list):
        self.model.eval()
        
        with torch.no_grad():
            dataset = TestDataset(test_list)
            test_loader = DataLoader(dataset, batch_size=128, shuffle=False)
            print ("==============>test_list.len:", len(test_list)) 
            print ("==============>dataset:", len(dataset)) 
            scores = []
            for i, (slice) in enumerate(test_loader):
                input_data = slice.cuda()
                input_data = input_data.unsqueeze(1) #增加通道
                out = self.model(input_data)
                out = out.squeeze(-1).float()
                if (len(out.shape) > 0):
                    # TypeError: iteration over a 0-d tensor
                    for item in out:
                        scores.append(item.cpu().item())
            print ("==============>scores.len:", len(scores))        
        return scores



def main(file_no):
    set_seed(file_no)

    all_data, split_pos, anomaly_range = get_series(file_no)

    all_data = minmax_scale(all_data)

    #all_data = aggregate(all_data, win_size=10)

    train_data, test_data = all_data[:split_pos], all_data[split_pos:]

    train_pos_list = create_window_list(train_data, 128)

    start_time = time.time()
    train_neg_data = []
    for train_pos in train_pos_list:
        augamented_neg_list = augament(train_pos)
        for augamented_neg in augamented_neg_list:
            train_neg_data.append(augamented_neg)
    print("aug_time:", (time.time() - start_time))

    channel = Channel(file_no, train_pos_list, train_neg_data)
    channel.train()

    test_list = create_window_list(test_data, 128)
    scores = channel.predict(test_list=test_list)

    correct_range = (anomaly_range[0]-100, anomaly_range[1]+100)
    pos = np.argmin(scores) + len(train_data)
    if pos >= correct_range[0] and pos <= correct_range[1]:
        return 1
    else:
        return -1

if __name__ == "__main__":
    correct_cnt = 0
    error_cnt = 0

    ret = None
    for i in range(1,251):
        if i in [239,240,241]:
            ret = -1
        else:
            ret = main(i)

        if (ret > 0):
            correct_cnt += 1
            print (f"({i}) correct, ==========>correct_cnt:{correct_cnt}, error_cnt:{error_cnt}")
        else:
            error_cnt += 1
            print (f"({i}) error, ============>correct_cnt:{correct_cnt}, error_cnt:{error_cnt}")