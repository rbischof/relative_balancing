import os
import numpy as np
import tensorflow as tf

from utils import show_image

TOL = 1e-5

class MNIST():
    def __init__(self, a:float=10, backward:bool=False):
        self.a = a
        self.backward = backward
        self.num_b_losses = 2 if not backward else 1
        self.x_train = np.load('data/x_train.npy')
        self.x_test = np.load('data/x_test.npy')

    def training_batch(self, batch_size=1024):
        ix = np.random.randint(0, len(self.x_train), batch_size)
        return self.x_train[ix], self.x_train[ix]

    def validation_batch(self, batch_size=4096):
        ix = np.random.randint(0, len(self.x_test), batch_size)
        return self.x_test[ix], self.x_test[ix], self.x_test[ix]

    @tf.function
    def calculate_loss(self, model, x, y, aggregate_boundaries=False, training=False):
        p = model[0](x, training=training)
        dpdx, dpdy = tf.image.image_gradients(p)
        dxdx, dxdy = tf.image.image_gradients(x)
        
        if self.backward:
            f_loss = tf.reduce_mean((p - x)**2)
            b_loss = [tf.reduce_mean((model[1]*dpdx - self.a*dxdx)**2) + tf.reduce_mean((model[1]*dpdy - self.a*dxdy)**2)]
            return f_loss, b_loss
        else:
            f_loss = tf.reduce_mean((p - x)**2)
            b_loss = [self.a*tf.reduce_mean((dpdx - dxdx)**2), self.a*tf.reduce_mean((dpdy - dxdy)**2)]

            if aggregate_boundaries:
                return f_loss, [b_loss[0] + b_loss[1]]
            else:
                return f_loss, b_loss


    @tf.function
    def validation_loss(self, model, x, y, w):
        p = model[0](x, training=False)
        if not self.backward:
            return tf.reduce_mean((p - x)**2)
        else:
            return tf.reduce_mean((p - x)**2), tf.reduce_mean((model[1] - self.a)**2)


    def visualise(self, model:tf.keras.Model, path:str=None):
        x, _ = self.validation_batch()
        p, _ = model[0].predict(x[:1])

        show_image(p.reshape(28, 28), os.path.join(path, 'predicted'))
        show_image(x[0].numpy().reshape(28, 28), os.path.join(path, 'real'))
        show_image((x[0].numpy().reshape(28, 28) - p.reshape(28, 28))**2, os.path.join(path, 'squared_error'))