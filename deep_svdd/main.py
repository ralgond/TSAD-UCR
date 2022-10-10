from common import set_seed
from ucr_dataset import get_data
from network import AE_Conv_128, AE_Conv_192
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common import UCRDataset, UCRDatasetForTest, Sigmoid, Interpolation, set_seed
import torch.optim as optim
import matplotlib.pyplot as plt
from ae_trainer import train
from deep_svdd_trainer import DeepSVDDTrainer
import numpy as np

def main1(train_data, test_data, WIN_SIZE=128):
    ae = None
    if WIN_SIZE == 128:
        ae = AE_Conv_128()
    elif WIN_SIZE == 192:
        ae = AE_Conv_192()
    else:
        raise NotImplementedError()

    loader_for_train = DataLoader(UCRDataset(train_data, WIN_SIZE), batch_size=512, shuffle=False)
    optimizer = optim.Adam(ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    ae = train(train_data, test_data, ae, loader_for_train, optimizer, criterion, add_channel=True, n_epochs=5)

    deep_svdd_trainer = DeepSVDDTrainer('soft-boundary', 0, None, 0.1, n_epochs=15)
    deep_svdd_trainer.train(loader_for_train, ae.encoder)

    loader_for_test = DataLoader(UCRDataset(test_data, WIN_SIZE), batch_size=512, shuffle=False)
    scores = deep_svdd_trainer.test(loader_for_test, ae.encoder)

    return scores

    # loader_for_test = DataLoader(UCRDatasetForTest(train_data, 128), batch_size=100)

    # logits_data = []

    # ae.eval()
    # with torch.no_grad():
    #     for data in loader_for_test:
    #         data = data.unsqueeze(1)
    #         data = data.to("cuda")
    #         logits = ae(data)
    #         logits_data.extend(logits.cpu().view(-1))
    # print("===========>len(train_data):", len(train_data))
    # print("===========>len(logits_data):", len(logits_data))
    # plt.figure(figsize=(16,9))
    # plt.plot([i for i in range(len(train_data))], train_data)
    # plt.plot([i for i in range(len(logits_data))], logits_data)
    # plt.show()

    

    

def main(file_no:int):
    set_seed(42)

    WIN_SIZE = 192

    train_data, test_data, anomaly_range = get_data(file_no)

    scores = main1(train_data, test_data, WIN_SIZE)
    
    correct_range = (anomaly_range[0]-100, anomaly_range[1]+100)

    pred_pos = np.argmax(scores) + len(train_data)

    if (pred_pos >= correct_range[0] and pred_pos <= correct_range[1]):
        return 1
    else:
        return -1

if __name__ == "__main__":
    correct_cnt = 0
    error_cnt = 0

    for i in range(1,26):
        ret = main(i)

        if (ret > 0):
            correct_cnt += 1
            print (f"({i}) correct, ==========>correct_cnt:{correct_cnt}, error_cnt:{error_cnt}")
        else:
            error_cnt += 1
            print (f"({i}) error, ============>correct_cnt:{correct_cnt}, error_cnt:{error_cnt}")