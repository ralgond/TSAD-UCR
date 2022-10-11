import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ucr_dataset import load_data

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

WIN_SIZE = 128
latent_dim = 2

encoder_inputs = keras.Input(shape=(WIN_SIZE, 1))
x = layers.Conv1D(8, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv1D(16, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
#encoder.summary()


latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(32 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((32, 64))(x)
x = layers.Conv1DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv1DTranspose(8, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
#decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(0,1)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        return self.decoder(z)

def main(i):
    X_train, X_test, all_data, split_pos, anomaly_range = load_data(i, WIN_SIZE)

    print ("++++++++++++",X_train.shape)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(X_train, epochs=5, shuffle=False, batch_size=128)

    X_pred = vae.predict(X_test)
    #print("++++++++++++++result.shape:", X_pred.shape)

    X_score = []
    for test, pred in zip(X_test, X_pred):
        score = mean_absolute_error(test, pred)
        X_score.append(score)

    correct_range = (anomaly_range[0]-100, anomaly_range[1]+100)
    pos = np.argmax(X_score) + split_pos
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


# (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
# mnist_digits = np.concatenate([x_train, x_test], axis=0)
# mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

# vae = VAE(encoder, decoder)
# vae.compile(optimizer=keras.optimizers.Adam())
# vae.fit(mnist_digits, epochs=30, batch_size=128)