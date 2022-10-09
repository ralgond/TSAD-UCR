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

def main(file_no:int):

    SEQ_LEN = 128

    all_data, train_test_split_pos, abnormal_range = get_series(file_no)

    moving_avg_all_data2 = np.array(all_data, dtype=np.float32)
    moving_avg_all_data2 = (moving_avg_all_data2 - np.min(moving_avg_all_data2)) / (np.max(moving_avg_all_data2) - np.min(moving_avg_all_data2))


    train_series, test_series = moving_avg_all_data2[:train_test_split_pos], moving_avg_all_data2[train_test_split_pos:]

    dataset_for_pretrain = Dataset(train_series, SEQ_LEN) # skip SEQ_LEN nan
    x_train = []    
    for i in range(len(dataset_for_pretrain)):
        x_train.append(dataset_for_pretrain[i])
    x_train = shuffle_x_train(x_train)
    x_train_tensor = tf.constant(x_train, dtype=tf.float32)

    encoder, autoencoder, history = Lukas_pretrain(x_train=x_train_tensor, seq_len=SEQ_LEN)


    