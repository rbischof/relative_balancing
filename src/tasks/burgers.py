import os
import scipy.io
import numpy as np
import tensorflow as tf

from utils import show_image

TOL = 1e-5

class Burgers():
    def __init__(self, inverse_var:float, inverse:bool):
        self.inverse = inverse
        self.nue = inverse_var if inverse_var is not None else 0.01/np.pi
        self.num_b_losses = 3 if not inverse else 1
        data = scipy.io.loadmat('data/burgers_shock_mu_01_pi.mat')  	          # Load data from file
        self.x = data['x']                                                        # 256 points between -1 and 1 [256x1]
        self.t = data['t']                                                        # 100 time points between 0 and 1 [100x1] 
        self.u = tf.cast(np.rot90(data['usol'], k=1)[::-1], dtype=tf.float32)     # solution of 256x100 grid points

    def training_batch(self, batch_size=1024):
        if not self.inverse:
            x_in = tf.random.uniform((2*batch_size//3, 1), minval=-1, maxval=1, dtype=tf.float32)
            x_b1 = tf.random.uniform((batch_size//9, 1), minval=-1, maxval=(-1+TOL), dtype=tf.float32)
            x_b2 = tf.random.uniform((batch_size//9, 1), minval=(1-TOL), maxval=1, dtype=tf.float32)
            x_b3 = tf.random.uniform((batch_size//9, 1), minval=-1, maxval=1, dtype=tf.float32)
            x = tf.concat([x_in, x_b1, x_b2, x_b3], axis=0)

            t_in = tf.random.uniform((2*batch_size//3, 1), minval=0, maxval=1, dtype=tf.float32)
            t_b1 = tf.random.uniform((batch_size//9, 1), minval=0, maxval=1, dtype=tf.float32)
            t_b2 = tf.random.uniform((batch_size//9, 1), minval=0, maxval=1, dtype=tf.float32)
            t_b3 = tf.random.uniform((batch_size//9, 1), minval=0, maxval=TOL, dtype=tf.float32)
            t = tf.concat([t_in, t_b1, t_b2, t_b3], axis=0)
        else:
            ixx, ixt = np.random.randint(0, 256, batch_size), np.random.randint(0, 100, batch_size)
            x = tf.cast(self.x.flatten()[ixx].reshape((batch_size, 1)), dtype=tf.float32)
            t = tf.cast(self.t.flatten()[ixt].reshape((batch_size, 1)), dtype=tf.float32)

        return x, t
    
    def validation_batch(self):
        x, t = np.meshgrid(self.x, self.t)
        return tf.cast(tf.reshape(x, (-1, 1)), dtype=tf.float32), tf.cast(tf.reshape(t, (-1, 1)), dtype=tf.float32), tf.reshape(self.u, (-1, 1))


    @tf.function
    def calculate_loss(self, model, x, t, aggregate_boundaries=False, training=False):
        if self.inverse:
            x, t, u = self.validation_batch()
        # predictions and derivatives
        u_pred = model[0](tf.concat([x, t], axis=-1), training=training)
        du_dx, du_dt = tf.gradients(u_pred, [x, t])
        du_dxx = tf.gradients(du_dx, x)[0]

        if self.inverse:
            f_loss = tf.reduce_mean((du_dt + u_pred*du_dx - model[1]*du_dxx)**2)
            u_loss = tf.reduce_mean((u_pred - u)**2)
            return f_loss, [u_loss]
        else:
            # governing equation loss
            f_loss = tf.reduce_mean((du_dt + u_pred*du_dx - self.nue*du_dxx)**2)

            # boundary conditions loss
            xl = tf.cast(x < (-1 + TOL), dtype=tf.float32)
            xu = tf.cast(x > ( 1 - TOL), dtype=tf.float32)
            tl = tf.cast(t < TOL, dtype=tf.float32)

            if aggregate_boundaries:
                b_loss = tf.reduce_mean((u_pred * (xl + xu + tl))**2)
                return f_loss, [b_loss]
            else:
                b1_loss = tf.reduce_mean((u_pred * xu)**2)
                b2_loss = tf.reduce_mean((u_pred * xl)**2)
                b3_loss = tf.reduce_mean(((-tf.math.sin(np.pi*x) - u_pred) * tl)**2)
                return f_loss, [b1_loss, b2_loss, b3_loss]
    
    @tf.function
    def validation_loss(self, model, x, t, u):
        u_pred = model[0](tf.concat([x, t], axis=-1), training=False)
        if not self.inverse:
            return tf.reduce_mean((u - u_pred)**2)
        else:
            return tf.reduce_mean((u - u_pred)**2), tf.reduce_mean((model[1] - self.nue)**2)

    def visualise(self, model:tf.keras.Model, path:str=None):
        x, t, u = self.validation_batch()
        u_pred = model[0].predict(tf.concat([x, t], axis=-1))

        show_image(u_pred.reshape(32, 32), os.path.join(path, 'u_predicted'), extent=[-1, 1, 0, 1])
        show_image(u.numpy().reshape(32, 32), os.path.join(path, 'u_real'), extent=[-1, 1, 0, 1])
        show_image((u.numpy().reshape(32, 32) - u_pred.reshape(32, 32))**2, os.path.join(path, 'u_squared_error'), extent=[-1, 1, 0, 1])