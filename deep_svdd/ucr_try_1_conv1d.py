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
from ucr_utils_data import Dataset, DataLoader, XTrainDataLoader
from ucr_debug import get_series_with_abnormal_1
from tensorflow.keras.layers import Conv1D, MaxPool1D,Input, UpSampling1D, Dense, Flatten, Reshape, Dropout, BatchNormalization,ReLU,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Ones

def main():
    all_data, train_test_split_pos, abnormal_range = get_series(251)

    train_series, test_series = all_data[:train_test_split_pos], all_data[train_test_split_pos:]

    dataset_for_pretrain = Dataset(train_series, 2)
    x_train = []
    for i in range(len(dataset_for_pretrain)):
        x_train.append(dataset_for_pretrain[i])

    print (x_train)
    x_train_tensor = tf.constant(x_train, dtype=tf.float32)
    x_train_tensor2 = tf.reshape(x_train_tensor, shape=(-1, x_train_tensor.shape[0], x_train_tensor.shape[1]))

    input = Input(shape=(None, 2))
    output = Conv1D(3, 3, use_bias=False, kernel_initializer=Ones())(input)

    model = Model(inputs=input, outputs=output)
    y = model(x_train_tensor2)

    print (y)

if __name__ == "__main__":
    main()