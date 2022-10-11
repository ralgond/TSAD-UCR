import os
import numpy as np
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

import tensorflow as tf
import tensorflow.keras as keras
from ucr_dataset import get_series
from ucr_autoencoder_conv import Conv_AE_pretrain
from ucr_autoencoder_vanilla import Vanilla_AE_pretrain
from ucr_autoencoder_lukas import Lukas_pretrain
from ucr_deepsvdd import UCRDeepSVDD
from ucr_utils_data import Dataset, DataLoader, XTrainDataLoader, shuffle_x_train
from ucr_debug import get_series_with_abnormal_1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main(file_no:int):
    set_seed(42)

    SEQ_LEN = 128

    all_data, train_test_split_pos, abnormal_range = get_series(file_no)

    moving_avg_all_data2 = np.array(all_data, dtype=np.float32)
    moving_avg_all_data2 = (moving_avg_all_data2 - np.min(moving_avg_all_data2)) / (np.max(moving_avg_all_data2) - np.min(moving_avg_all_data2))


    train_series, test_series = moving_avg_all_data2[:train_test_split_pos], moving_avg_all_data2[train_test_split_pos:]

    #train_series, test_series, abnormal_range = get_series_with_abnormal_1()

    dataset_for_pretrain = Dataset(train_series, SEQ_LEN) # skip SEQ_LEN nan
    x_train = []    
    for i in range(len(dataset_for_pretrain)):
        x_train.append(dataset_for_pretrain[i])
    x_train = shuffle_x_train(x_train)
    x_train_tensor = tf.constant(x_train, dtype=tf.float32)

    encoder, autoencoder, history = Lukas_pretrain(x_train=x_train_tensor, seq_len=SEQ_LEN)

    deepSVDD = UCRDeepSVDD(encoder=encoder, objective='one-class', win_size=SEQ_LEN)

    train_loader = XTrainDataLoader(x_train, batch_size=256)

    deepSVDD.fit(train_loader, n_epochs=30)

    score_list = deepSVDD.predict(DataLoader(Dataset(test_series, SEQ_LEN)))

    print (np.argmax(score_list))

    correct_range = (abnormal_range[0]-100, abnormal_range[1]+100)
    return correct_range, len(train_series)+np.argmax(score_list)

if __name__ == "__main__":
    error_file = open("./error_pos.txt", "w+")
    correct_count = 0
    error_count = 0
    for file_no in range(1,251):
        correct_range, predict_no = main(file_no)
        print(correct_range, predict_no)
        if predict_no >= correct_range[0] and predict_no <= correct_range[1]:
            correct_count += 1
        else:
            error_count += 1
            error_file.write(f"{file_no} ({correct_range[0]},{correct_range[1]}) {predict_no}\n")
        print("=======================>correct_count:",correct_count)
        print("=======================>error_count:",error_count)
    error_file.close()



