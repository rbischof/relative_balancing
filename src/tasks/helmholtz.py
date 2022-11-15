import os
import numpy as np
import tensorflow as tf

from utils import show_image

TOL = 1e-5

class Helmholtz():
    def __init__(self, inverse:bool=False, inverse_var:float=None):
        self.inverse = inverse
        self.num_b_losses = 4 if not inverse else 1
        self.k = inverse_var if inverse_var is not None else 1
        self.data_min = (-1, -1)
        self.data_max = (1, 1)

    def training_batch(self, batch_size:int=1024):
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
        x, y = np.mgrid[-1:1:complex(0, 32), -1:1:complex(0, 32)]
        x, y = tf.cast(x.reshape(1024, 1), dtype=tf.float32), \
                        tf.cast(y.reshape(1024, 1), dtype=tf.float32)
        u = tf.cast(tf.math.sin(np.pi*x)*tf.math.sin(4*np.pi*y), dtype=tf.float32)
        return x, y, u

    def calculate_loss(self, model:tf.keras.Model, x, y, aggregate_boundaries:bool=False, training:bool=False):
        # predictions and derivatives
        u_pred = model[0]([x, y], training=training)
        du_dx, du_dy = tf.gradients(u_pred, [x, y])
        du_dxx = tf.gradients(du_dx, x)[0]
        du_dyy = tf.gradients(du_dy, y)[0]
        f_pred = du_dxx + du_dyy + self.k**2*u_pred

        sin_xy = tf.math.sin(np.pi*x)*tf.math.sin(4*np.pi*y)
        if self.inverse:
            f_loss = tf.reduce_mean(((-np.pi**2 - (4*np.pi)**2 + model[1]**2) * sin_xy - f_pred)**2)
            u_loss = tf.reduce_mean((self.k**2*sin_xy - model[1]**2*u_pred)**2)
            return f_loss, [u_loss]
        else:
            f_loss = tf.reduce_mean((f_pred - (-np.pi**2 - (4*np.pi)**2 + self.k**2) * sin_xy)**2)

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
    def validation_loss(self, model:tf.keras.Model, x, y, u):
        u_pred = model[0]([x, y], training=False)
        if not self.inverse:
            return tf.reduce_mean((u - u_pred)**2)
        else:
            return tf.reduce_mean((u - u_pred)**2), tf.reduce_mean((model[1] - self.k)**2)


    def visualise(self, model:tf.keras.Model, path:str=None):
        x, y, u = self.validation_batch()
        u_pred = model[0].predict([x, y])

        show_image(u_pred.reshape(32, 32), os.path.join(path, 'u_predicted'), extent=[-1, 1, -1, 1])
        show_image(u.numpy().reshape(32, 32), os.path.join(path, 'u_real'), extent=[-1, 1, -1, 1])
        show_image((u.numpy().reshape(32, 32) - u_pred.reshape(32, 32))**2, os.path.join(path, 'u_squared_error'), extent=[-1, 1, -1, 1], format='%.1e')