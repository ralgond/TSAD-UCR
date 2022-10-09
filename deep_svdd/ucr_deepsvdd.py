import numpy as np
import time

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class UCRDeepSVDD:
    def __init__(self, encoder:Model, objective='one-class', nu:float=0.1,
                win_size:int=32, lr:float=1e-4) -> None:
        assert (win_size % 16 == 0)
        self.represetation_dim = int(win_size / 16)
        self.objective = objective
        self.encoder = encoder
        self.nu = nu

        self.R = tf.constant(0, dtype=tf.float32) # radius R initialized with 0 by default.
        self.c = tf.zeros(self.represetation_dim, dtype=tf.float32)

        self.warm_up_n_epochs = 5

        self.optimizer = Adam(learning_rate=1e-4)


    #@tf.function
    def __loss_fun(self, outputs):
        dist = tf.reduce_sum(tf.square(outputs - self.c), axis=-1)
        if self.objective == 'soft-boundary':
            scores = dist - self.R ** 2
            penalty = tf.reduce_mean(tf.maximum(scores, tf.zeros_like(scores)))
            loss = self.R ** 2 + (1 / self.nu) * penalty
            return loss, dist
        else:
            return tf.reduce_mean(dist), dist

    @tf.function
    def __train_step(self, x):
        with tf.GradientTape() as tape:
            logits = self.encoder(x, training=True)
            loss_value, dist = self.__loss_fun(logits)
        grads = tape.gradient(loss_value, self.encoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights))
        return loss_value, dist


    def fit(self, x_train_loader, n_epochs=15):
        
        self._init_c(x_train_loader)

        for epoch in range(n_epochs):
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            x_train_loader.shuffle()
            for x_batch_train in x_train_loader:
                loss_value, dist = self.__train_step(x_batch_train)
                loss_epoch += loss_value
                n_batches += 1

            # Update hypersphere radius R on mini-batch distances
            if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                self.R = tf.constant(self._get_R(dist, self.nu), dtype=tf.float32)

            epoch_train_time = time.time() - epoch_start_time
            print ('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch + 1, n_epochs, epoch_train_time, loss_epoch / n_batches))


    def predict(self, test_dataset):
        scores = list()

        for x_batch_test in test_dataset:
            logits = self.encoder(x_batch_test, training=False)
            s_batch = tf.reduce_sum(tf.square(logits - self.c), axis=-1)
            scores.append(s_batch)

        return np.concatenate(scores)


    def _init_c(self, x_train_loader, eps=1e-1):
        x_train_loader.shuffle()
        n_samples = 0

        latent_sum = np.zeros(self.represetation_dim)
        for x_batch_train in x_train_loader:
            #print("======================>...,", x_batch_train.shape, ", ", x_batch_train[0][0])
            latent_v = self.encoder(x_batch_train, training=False)
            n_samples += latent_v.shape.as_list()[0]
            latent_sum += tf.reduce_sum(latent_v, axis=0)
        
        c = latent_sum / n_samples

        c = c.numpy()

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.c = tf.constant(c)

    def _get_R(self, dist, nu):
        return np.quantile(np.sqrt(dist), 1 - nu)
