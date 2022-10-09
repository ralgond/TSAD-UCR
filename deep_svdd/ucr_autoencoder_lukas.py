"""
Lukasruff Autoencoder.

https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/networks/mnist_LeNet.py
"""
import os
from random import seed
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPool1D,Input, UpSampling1D, Dense, Flatten, Reshape, Dropout, BatchNormalization,ReLU,LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
import tensorflow.keras as keras
from ucr_dataset import get_series
import matplotlib.pyplot as plt

def Lukas_pretrain(x_train, seq_len: int):
    CHANNEL_1 = 16
    CHANNEL_2 = 32

    CHANNEL_OUTPUT = 1
    EPOCHS = 5
    BATCH_SIZE = 256

    KERNEL_SIZE = 15

    L2_REG = 0.005
    L1_REG = 0.02

    # input placeholder
    input = Input(shape=(seq_len, 1))

    # encoding layers
    x = Conv1D(CHANNEL_1, KERNEL_SIZE, padding="same", 
                kernel_initializer=initializers.RandomNormal(seed=0), use_bias=False)(input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(4, padding='same')(x)

    x = Conv1D(CHANNEL_2, KERNEL_SIZE, padding='same',
                kernel_initializer=initializers.RandomNormal(seed=0), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool1D(4, padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(int(seq_len/16), use_bias=False,
                    kernel_initializer=initializers.RandomNormal(seed=0))(x)

    # decoding layers
    x = Reshape((int(seq_len/16), 1))(encoded)

    # x = LeakyReLU()(x)
    # x = UpSampling1D(2)(x)

    x = UpSampling1D(4)(x)
    x = Conv1D(CHANNEL_2, KERNEL_SIZE, padding='same',
                kernel_initializer=initializers.RandomNormal(seed=0), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    

    x = UpSampling1D(4)(x)
    x = Conv1D(CHANNEL_1, KERNEL_SIZE, padding='same',
                kernel_initializer=initializers.RandomNormal(seed=0), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    

    decoded = Conv1D(CHANNEL_OUTPUT, KERNEL_SIZE, padding='same', activation='sigmoid',
                kernel_initializer=initializers.RandomNormal(seed=0), use_bias=False)(x)

    autoencoder = Model(inputs=input, outputs=decoded)
    encoder = Model(inputs=input, outputs=encoded)

    #autoencoder.summary()

    autoencoder.compile(optimizer=RMSprop(), loss='mse', metrics=['mse'])

    history_record = autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False)

    return encoder, autoencoder, history_record