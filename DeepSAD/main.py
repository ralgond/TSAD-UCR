from ucr_dataset import get_series
import numpy as np
import time


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from common import set_seed, SemiTrainDataset, TestDataset, minmax_scale, create_window_list, augament, aggregate
from network_lukas import LukasAE, LukasEncoder


class Channel:
    def __init__(self, id, pos_samples, neg_samples) -> None:
        self.id = id
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.eta = 1.0
        self.c = None  # hypersphere center c

        # Optimization parameters
        self.eps = 1e-6

        self.ae_net = LukasAE(128)
        self.ae_net.cuda()

        self.net = LukasEncoder(128)
        self.net.cuda()

    def pretrain(self):
        # Set loss
        criterion = nn.MSELoss(reduction='none')

        dataset = SemiTrainDataset(self.pos_samples, self.neg_samples)
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        optimizer = torch.optim.Adam(self.ae_net.parameters(), lr=0.0005)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(), gamma=0.1)

        n_epochs = 5

        self.ae_net.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _ = data
                inputs = inputs.cuda()

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                #print ("===============>inputs.shape:", inputs.shape)
                rec = self.ae_net(inputs.unsqueeze(1))
                rec = rec.squeeze(1)
                #print ("===============>rec.shape:", rec.shape)
                rec_loss = criterion(rec, inputs)
                loss = torch.mean(rec_loss)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            #scheduler.step()
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print(f'| Epoch: {epoch + 1:03}/{n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def train(self):
        dataset = SemiTrainDataset(self.pos_samples, self.neg_samples)
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0005)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(), gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(DataLoader(dataset, batch_size=128, shuffle=False), self.net)
            print('Center c initialized.')
            print(self.c)

        print (f"==========>pos_samples.len: {len(self.pos_samples)}, neg_samples.len: {len(self.neg_samples)}")

        
        self.net.train()

        for epoch in range(10):
            
            loss_epoch = 0.
            batch_epoch = 0
            for i, (slices, semi_targets) in enumerate(train_loader):
                optimizer.zero_grad()

                input_data, semi_targets = slices.cuda(), semi_targets.cuda()
                input_data = input_data.unsqueeze(1) #增加通道
                outputs = self.net(input_data)

                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()

                batch_epoch += 1
                loss_epoch += loss.cpu().item()

            #scheduler.step()
            print (f"epoch:{epoch}, loss:{loss_epoch/batch_epoch:.5f}")
            
    def predict(self, test_list):
        self.net.eval()
        
        with torch.no_grad():
            dataset = TestDataset(test_list)
            test_loader = DataLoader(dataset, batch_size=128, shuffle=False)
            print ("==============>test_list.len:", len(test_list)) 
            print ("==============>dataset:", len(dataset)) 
            scores = []
            for i, (slices) in enumerate(test_loader):
                input_data = slices.cuda()
                input_data = input_data.unsqueeze(1) #增加通道
                outputs = self.net(input_data)
                outputs = outputs.squeeze(-1).float()

                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                scores.extend(dist.cpu())
            print ("==============>scores.len:", len(scores))        
        return scores

    def init_center_c(self, train_loader: DataLoader, net: nn.Module, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device="cuda")

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.to("cuda")
                outputs = net(inputs.unsqueeze(1))
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples
        #print(c)

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def main(file_no):
    set_seed(file_no)

    all_data, split_pos, anomaly_range = get_series(file_no)

    all_data = minmax_scale(all_data)

    #all_data = aggregate(all_data, win_size=5)

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
    channel.pretrain()

    channel.train()

    test_list = create_window_list(test_data, 128)
    scores = channel.predict(test_list=test_list)

    correct_range = (anomaly_range[0]-100, anomaly_range[1]+100)
    pos = np.argmax(scores) + len(train_data)
    if pos >= correct_range[0] and pos <= correct_range[1]:
        return 1
    else:
        return -1

if __name__ == "__main__":
    correct_cnt = 0
    error_cnt = 0

    ret = None
    for i in range(1, 251):
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