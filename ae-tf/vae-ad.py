from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.losses import mse, binary_crossentropy, kl_divergence
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from ucr_dataset import get_series
from slide_window import create_window_list

def main(file_no:int):
    all_data, split_pos, anomaly_range = get_series(file_no)
    train_data, test_data = np.array(all_data[:split_pos]), np.array(all_data[split_pos:])

    scaler = MinMaxScaler()
    scaler.fit(train_data.reshape(-1,1))

    train_data = scaler.transform(train_data.reshape(-1,1)).squeeze(1)
    test_data = scaler.transform(test_data.reshape(-1,1)).squeeze(1)

    # print ("============>train_data.shape:", train_data.shape)
    # The reparameterization trick

    def sample(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    WIN_SIZE = 128

    original_dim = WIN_SIZE
    input_shape = (original_dim,)
    intermediate_dim = int(original_dim / 2)
    latent_dim = int(original_dim / 3)

    # encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    # use the reparameterization trick and get the output from the sample() function
    z = Lambda(sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, z, name='encoder')
    #encoder.summary()

    # decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    # Instantiate the decoder model:
    decoder = Model(latent_inputs, outputs, name='decoder')
    #decoder.summary()

    outputs = decoder(encoder(inputs))
    vae_model = Model(inputs, outputs, name='vae_mlp')

    # the KL loss function:
    def vae_loss(x, x_decoded_mean):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        reconstruction_loss = K.sum(K.square(x - x_decoded_mean))
        # compute the KL loss
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.square(K.exp(z_log_var)), axis=-1)
        # return the average loss over all 
        total_loss = K.mean(reconstruction_loss + kl_loss)    
        #total_loss = reconstruction_loss + kl_loss
        return total_loss

    opt = optimizers.Adam(learning_rate=0.0001, clipvalue=0.5)
    vae_model.compile(optimizer=opt, loss=vae_loss)
    #vae_model.summary()


    # Finally, we train the model:
    X_train = create_window_list(train_data, WIN_SIZE)
    X_train = np.array(X_train)

    results = vae_model.fit(X_train, X_train,
                            shuffle=False,
                            epochs=32,
                            batch_size=256)


    # Predict test, found the anomalies
    X_test = create_window_list(test_data, WIN_SIZE)
    X_test = np.array(X_test)


    X_pred = vae_model.predict(X_test)
    #print("++++++++++++++result.shape:", X_pred.shape)

    X_score = []
    for test, pred in zip(X_test, X_pred):
        score = mean_absolute_error(test, pred)
        X_score.append(score)

    correct_range = (anomaly_range[0]-100, anomaly_range[1]+100)
    pos = np.argmax(X_score) + len(train_data)
    if pos >= correct_range[0] and pos <= correct_range[1]:
        return 1
    else:
        return -1

if __name__ == "__main__":
    correct_count = 0
    error_count = 0
    for i in range(1,251):
        ret = main(i)
        status = None
        if ret > 0:
            correct_count += 1
            status = "correct"
        else:
            error_count += 1
            status = "error"
        print (f"({i}) {status}=========> correct:{correct_count}, error:{error_count}")