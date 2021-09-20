import os
import numpy as np
import tensorflow as tf

from utils import show_image

TOL = 1e-5

class Helmholtz():
    def __init__(self, k:float=1, backward:bool=False):
        self.k = k
        self.backward = backward
        self.num_b_losses = 4 if not backward else 1

    def training_batch(self, batch_size=1024):
        x_in = tf.random.uniform((2*batch_size//3, 1), minval=-1, maxval=1, dtype=tf.float32)
        x_b1 = tf.random.uniform((batch_size//12, 1), minval=-1, maxval=(-1+TOL), dtype=tf.float32)
        x_b2 = tf.random.uniform((batch_size//12, 1), minval=(1-TOL), maxval=1, dtype=tf.float32)
        x_b3 = tf.random.uniform((batch_size//12, 1), minval=-1, maxval=1, dtype=tf.float32)
        x_b4 = tf.random.uniform((batch_size//12, 1), minval=-1, maxval=1, dtype=tf.float32)
        x = tf.concat([x_in, x_b1, x_b2, x_b3, x_b4], axis=0)

        y_in = tf.random.uniform((2*batch_size//3, 1), minval=-1, maxval=1, dtype=tf.float32)
        y_b1 = tf.random.uniform((batch_size//12, 1), minval=-1, maxval=1, dtype=tf.float32)
        y_b2 = tf.random.uniform((batch_size//12, 1), minval=-1, maxval=1, dtype=tf.float32)
        y_b3 = tf.random.uniform((batch_size//12, 1), minval=-1, maxval=(-1+TOL), dtype=tf.float32)
        y_b4 = tf.random.uniform((batch_size//12, 1), minval=(1-TOL), maxval=1, dtype=tf.float32)
        y = tf.concat([y_in, y_b1, y_b2, y_b3, y_b4], axis=0)

        return x, y

    def validation_batch(self):
        x, y = self.training_batch()
        u = tf.cast(tf.math.sin(np.pi*x)*tf.math.sin(4*np.pi*y), dtype=tf.float32)
        return x, y, u

    def calculate_loss(self, model, x, y, aggregate_boundaries=False, training=False):
        # predictions and derivatives
        u_pred = model[0](tf.concat([x, y], axis=-1), training=training)
        du_dx, du_dy = tf.gradients(u_pred, [x, y])
        du_dxx = tf.gradients(du_dx, x)[0]
        du_dyy = tf.gradients(du_dy, y)[0]
        f_pred = du_dxx + du_dyy + self.k*u_pred

        sin_xy = tf.math.sin(np.pi*x)*tf.math.sin(4*np.pi*y)
        if self.backward:
            f_loss = tf.reduce_mean((np.pi**2*sin_xy - (4*np.pi)**2*sin_xy + model[1]*sin_xy - f_pred)**2)
            u_loss = tf.reduce_mean((tf.math.sin(np.pi*x)*tf.math.sin(4*np.pi*y) - u_pred)**2)
            return f_loss, [u_loss]
        else:
            f_loss = tf.reduce_mean((f_pred + np.pi**2*sin_xy + (4*np.pi)**2*sin_xy - sin_xy)**2)

            # boundary conditions loss
            xl = tf.cast(x < (-1 + TOL), dtype=tf.float32)
            xu = tf.cast(x > ( 1 - TOL), dtype=tf.float32)
            yl = tf.cast(y < (-1 + TOL), dtype=tf.float32)
            yu = tf.cast(y > ( 1 - TOL), dtype=tf.float32)

            if aggregate_boundaries:
                b_loss = tf.reduce_mean((u_pred * (xl + xu + yl + yu))**2)
                return f_loss, [b_loss]
            else:
                b1_loss = tf.reduce_mean((u_pred * xl)**2)
                b2_loss = tf.reduce_mean((u_pred * xu)**2)
                b3_loss = tf.reduce_mean((u_pred * yl)**2)
                b4_loss = tf.reduce_mean((u_pred * yu)**2)
                return f_loss, [b1_loss, b2_loss, b3_loss, b4_loss]

    @tf.function
    def validation_loss(self, model, x, y, u):
        u_pred = model[0](tf.concat([x, y], axis=-1), training=False)
        if not self.backward:
            return tf.reduce_mean((u - u_pred)**2)
        else:
            return tf.reduce_mean((u - u_pred)**2), tf.reduce_mean((model[1] - self.k)**2)


    def visualise(self, model:tf.keras.Model, path:str=None):
        x, y, u = self.validation_batch()
        u_pred, _ = model[0].predict(tf.concat([x, y], axis=-1))

        show_image(u_pred.reshape(self.width, self.height), os.path.join(path, 'u_predicted'), extent=[-1, 1, 0, 1])
        show_image(u[0].numpy().reshape(self.width, self.height), os.path.join(path, 'u_real'), extent=[-1, 1, 0, 1])
        show_image((u[0].numpy().reshape(self.width, self.height) - u_pred[0].reshape(self.width, self.height))**2, os.path.join(path, 'u_squared_error'), extent=[-1, 1, 0, 1])