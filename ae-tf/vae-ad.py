from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Layer, Lambda, Dense, Conv1D, Flatten
from tensorflow.keras.losses import mse, binary_crossentropy, kl_divergence
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from ucr_dataset import get_series
from slide_window import create_window_list

# coefficients 
batch_size = 64
original_dim = 128
latent_dim = 16
intermediate_dim = 64
epochs = 1
epsilon_std = 1.0

'''
# loss function layer
'''
class VAE_loss(Layer):
  def __init__(self, **kwargs):
    self.is_placeholder = True
    super(VAE_loss, self).__init__(**kwargs)

  def vae_loss(sトロピー
    reconst_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean) 
    # 事前分布と事後分布のD_KLの値
    kl_loss = - 0.5 * K.sum(1 + K.log(K.square(z_sigma)) - K.square(z_mean) - K.square(z_sigma), axis=-1)
    return K.mean(reconst_loss + kl_loss)

  def call(self, inputs):
    x = inputs[0]
    x_decoded_mean = inputs[1]
    z_sigma = inputs[2]
    z_mean = inputs[3]
    loss = self.vae_loss(x, x_decoded_mean, z_sigma, z_mean)
    self.add_loss(loss, inputs=inputs)
    return x

class VAE(object):
    # save coefficients in advance
    # コンストラクタで定数を先に渡しておく
    def __init__(self, original_dim, latent_dim, intermediate_dim, batch_size, epsilon_std):
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.batch_size = batch_size
        self.epsilon_std = epsilon_std

    def encoder(self):
        x = Input(shape=(self.original_dim, ))

        #hidden = Dense(self.intermediate_dim, activation='relu')(x)
        hidden = Conv1D(16, 3, padding=1)(x)
        hidden = Flatten()(hidden)
        z_mean = Dense(self.latent_dim, activation='linear')(hidden)
        z_sigma = Dense(self.latent_dim, activation='linear')(hidden)

        return Model(x, [z_mean, z_sigma])

    def decoder(self):
        z_mean = Input(shape=(self.latent_dim, ))
        z_sigma = Input(shape=(self.latent_dim, ))
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_sigma])
        h_decoded = Dense(self.intermediate_dim, activation='relu')(z)
        x_decoded_mean = Dense(self.original_dim, activation='sigmoid')(h_decoded)

        return Model([z_mean, z_sigma], x_decoded_mean)
    
    # サンプル生成用デコーダ
    def generator(self, _decoder):
        decoder_input = Input(shape=(self.latent_dim,))
        _, _, _, decoder_dense1, decoder_dense2 = _decoder.layers
        h_decoded = decoder_dense1(decoder_input)
        x_decoded_mean = decoder_dense2(h_decoded)

        return Model(decoder_input, x_decoded_mean)

    def sampling(self, args):
        z_mean, z_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,
                              stddev=self.epsilon_std)
        return z_mean + K.exp(z_sigma / 2) * epsilon

    def build_vae(self, _encoder, _decoder):
        _, encoder_dense, encoder_mean, encoder_sigma = _encoder.layers

        x = Input(shape=(self.original_dim, ))
        hidden = encoder_dense(x)
        z_mean = encoder_mean(hidden)
        z_sigma = encoder_sigma(hidden)

        self.z_m = z_mean
        self.z_s = z_sigma

        _, _, decoder_lambda, decoder_dense1, decoder_dense2 = _decoder.layers
        z = decoder_lambda([z_mean, z_sigma])
        h_decoded = decoder_dense1(z)
        x_decoded_mean = decoder_dense2(h_decoded)
        # カスタマイズした損失関数を付加する訓練用レイヤー
        y = VAE_loss()([x, x_decoded_mean, z_sigma, z_mean])

        return Model(x, y)

    def model_compile(self, model):
        model.compile(optimizer='rmsprop', loss=None)
elf, x, x_decoded_mean, z_sigma, z_mean):
    # クロスエン
def main(file_no:int):
    all_data, split_pos, anomaly_range = get_series(file_no)
    train_data, test_data = np.array(all_data[:split_pos]), np.array(all_data[split_pos:])

    scaler = MinMaxScaler()
    scaler.fit(train_data.reshape(-1,1))

    train_data = scaler.transform(train_data.reshape(-1,1)).squeeze(1)
    test_data = scaler.transform(test_data.reshape(-1,1)).squeeze(1)

    WIN_SIZE = original_dim
    X_train = np.array(create_window_list(train_data, WIN_SIZE))
    X_test = np.array(create_window_list(test_data, WIN_SIZE))
    # # Finally, we train the model:
    # X_train_tmp = np.array(create_window_list(train_data, WIN_SIZE))
    # X_train = [np.expand_dims(train, 1) for train in X_train_tmp]

    # # Predict test, found the anomalies
    # X_test_tmp = np.array(create_window_list(test_data, WIN_SIZE))
    # X_test = [np.expand_dims(test, 1) for test in X_test_tmp]

    print (X_train.shape, X_test.shape)


    ''' 
    # Create an instance for the VAE model
    # VAEクラスからインスタンスを生成
    '''
    _vae = VAE(original_dim, latent_dim, intermediate_dim, batch_size, epsilon_std)
    _encoder = _vae.encoder()
    _decoder = _vae.decoder()

    _model = _vae.build_vae(_encoder, _decoder)
    _vae.model_compile(_model)
    #_model.summary()

    _hist = _model.fit(X_train,
        shuffle=False,
        epochs=epochs,
        batch_size=batch_size)

    # os._exit(0)





    X_pred = _model.predict(X_test)
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
    for i in range(1,25):
        ret = main(i)
        status = None
        if ret > 0:
            correct_count += 1
            status = "correct"
        else:
            error_count += 1
            status = "error"
        print (f"({i}) {status}=========> correct:{correct_count}, error:{error_count}")