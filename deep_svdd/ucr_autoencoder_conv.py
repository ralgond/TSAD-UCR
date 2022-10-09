"""
Convolutional Autoencoder.
"""
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPool1D,Input, UpSampling1D, Dense, Flatten, Reshape, Dropout, BatchNormalization,ReLU,LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import initializers
import tensorflow.keras as keras
from ucr_dataset import get_series
from ucr_utils_data import Dataset
import matplotlib.pyplot as plt

def Conv_AE_pretrain(x_train, seq_len: int):
    """
    build autoencoder.
    :param x_train:  the train data
    :return: encoder and decoder
    """
    
    CHANNEL_1 = 16
    CHANNEL_2 = 8
    CHANNEL_3 = 4

    CHANNEL_OUTPUT = 1
    EPOCHS = 3
    BATCH_SIZE = 128

    KERNEL_SIZE = 20

    # input placeholder
    input_image = Input(shape=(seq_len, 1))

    # encoding layers
    x = Conv1D(CHANNEL_1, KERNEL_SIZE, activation='relu', padding="same", 
                kernel_initializer=initializers.RandomNormal(seed=0), use_bias=False)(input_image)
    x = MaxPool1D(2, padding='same')(x)
    x = Conv1D(CHANNEL_2, KERNEL_SIZE, activation='relu', padding='same',
                kernel_initializer=initializers.RandomNormal(seed=0), use_bias=False)(x)
    x = MaxPool1D(2, padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(int(seq_len/4), use_bias=False)(x)


    # decoding layers
    x = Reshape((int(seq_len/4), 1))(encoded)
    x = Conv1D(CHANNEL_2, KERNEL_SIZE, activation='relu', padding='same',
                kernel_initializer=initializers.RandomNormal(seed=0), use_bias=False)(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(CHANNEL_1, KERNEL_SIZE,activation='relu', padding='same',
                kernel_initializer=initializers.RandomNormal(seed=0), use_bias=False)(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(CHANNEL_OUTPUT, 5, activation='sigmoid', padding='same',
                kernel_initializer=initializers.RandomNormal(seed=0), use_bias=False)(x)

    # build autoencoder, encoder, decoder
    autoencoder = Model(inputs=input_image, outputs=decoded)
    encoder = Model(inputs=input_image, outputs=encoded)

    # compile autoencoder
    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    autoencoder.compile(optimizer=RMSprop(), loss='mse', metrics=['mse'])

    # autoencoder.summary()

    # training
    # need return history, otherwise can not use history["acc"]
    history_record = autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, )

    return encoder, autoencoder, history_record


def plot_images(y_true, y_ae):
    plt.figure(figsize=(20, 4))
    x = np.arange(1, len(y_true)+1)
    plt.plot(x, y_true)
    plt.plot(x, y_ae)
    plt.show()



if __name__ == '__main__':

    SEQ_LEN = 128
    #SEQ_STRIDE = 1

    train_series, test_series, abnormal_range = get_series(1)

    dataset = Dataset(train_series, SEQ_LEN)

    x_train = []
    for i in range(len(dataset)):
        x_train.append(dataset[i])
    x_train = tf.constant(x_train, dtype=tf.float32)

    encoder, autoencoder, history_record = Conv_AE_pretrain(x_train=x_train, seq_len=SEQ_LEN)

    y_ae = autoencoder(x_train, training=False)[0]

    plot_images(dataset[0], y_ae)


