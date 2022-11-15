import os
import scipy.io
import numpy as np
import tensorflow as tf

from utils import show_image

TOL = 1e-5

class AllenCahn():
    def __init__(self, inverse:bool=False, inverse_var:float=None):
        self.inverse = inverse
        self.gamma = inverse_var if inverse_var else 0.0001
        self.num_b_losses = 3 if not inverse else 1
        self.data_min = (-1, 0)
        self.data_max = (1, 1)

        data = scipy.io.loadmat('data/allen_cahn.mat')
        x, t = data['x'], data['tt']
        self.u = tf.cast(data['uu'].reshape(-1, 1), dtype=tf.float32)
        x, t = np.meshgrid(x, t)
        self.x = tf.cast(tf.reshape(x.T, (-1, 1)), dtype=tf.float32)
        self.t = tf.cast(tf.reshape(t.T, (-1, 1)), dtype=tf.float32)

    def training_batch(self, batch_size:int=1024):
        step = batch_size//8

        x_dom = tf.random.uniform((4*step, 1), minval=-1.1, maxval=1.1, dtype=tf.float32)
        x_ful = tf.random.uniform((2*step, 1), minval=-1, maxval=1, dtype=tf.float32)
        x_bou = tf.random.uniform((step, 1), minval=-1, maxval=-1, dtype=tf.float32)
        x = tf.concat([x_dom, x_ful, x_bou, x_bou + 2 + TOL], axis=0)

        t_dom = tf.random.uniform((4*step, 1), minval=-0.1, maxval=1.1, dtype=tf.float32)
        t_low = tf.random.uniform((2*step, 1), minval=0, maxval=0, dtype=tf.float32)
        t_ful = tf.random.uniform((step, 1), minval=0, maxval=1, dtype=tf.float32)
        t = tf.concat([t_dom, t_low, t_ful, t_ful], axis=0)
        return x, t

    def validation_batch(self):
        return self.x, self.t, self.u

    @tf.function
    def calculate_loss(self, model:tf.keras.Model, x, t, aggregate_boundaries:bool=False, training:bool=False):
        if self.inverse:
            x, t, u = self.validation_batch()

        # predictions and derivatives
        batch_size = tf.shape(x)[0]
        step = batch_size//8
        u_pred = model[0]([x, t], training=training)
        du_dx, du_dt = tf.gradients(u_pred, [x, t])
        du_dxx = tf.gradients(du_dx, x)[0]

        if self.inverse:
            f_loss = tf.reduce_mean((1*((du_dt - tf.stop_gradient(model[1]*du_dxx) + tf.stop_gradient(5*(u_pred**3 - u_pred)))**2) +\
                    30*((tf.stop_gradient(du_dt) - model[1]*du_dxx + tf.stop_gradient(5*(u_pred**3 - u_pred)))**2) + \
                    0.1*((tf.stop_gradient(du_dt) - tf.stop_gradient(model[1]*du_dxx) + 5*(u_pred**3 - u_pred))**2)))
            u_loss = tf.reduce_mean((u_pred - u)**2)
            return f_loss, [u_loss]
        else:
            f_losses = tf.reduce_mean(1*((du_dt - tf.stop_gradient(self.gamma*du_dxx) + tf.stop_gradient(5*(u_pred**3 - u_pred)))**2) +\
                        50*((tf.stop_gradient(du_dt) - self.gamma*du_dxx + tf.stop_gradient(5*(u_pred**3 - u_pred)))**2) + \
                        0.1*((tf.stop_gradient(du_dt) - tf.stop_gradient(self.gamma*du_dxx) + 5*(u_pred**3 - u_pred))**2))

            # boundary conditions
            b_losses = [20*tf.reduce_mean((u_pred[4*step:6*step] - (x[4*step:6*step]**2*tf.math.cos(np.pi*x[4*step:6*step])))**2),
                        1*tf.reduce_mean((u_pred[6*step:7*step] - u_pred[7*step:])**2),
                        1*tf.reduce_mean((du_dx[6*step:7*step] - du_dx[7*step:])**2)]
            return f_losses, b_losses
    
    
    @tf.function
    def validation_loss(self, model:tf.keras.Model, x, t, u):
        u_pred = model[0]([x, t], training=False)
        if not self.inverse:
            return tf.reduce_mean((u - u_pred)**2)
        else:
            return tf.reduce_mean((u - u_pred)**2), tf.reduce_mean((model[1] - self.nue)**2)

    def visualise(self, model:tf.keras.Model, path:str=None):
        x, t, u = self.validation_batch()
        u_pred = model[0].predict([x, t])

        show_image(u_pred.reshape(512, 201), os.path.join(path, 'u_predicted'), extent=[0, 1, -1, 1], x_label='t', y_label='x')
        show_image(u.numpy().reshape(512, 201), os.path.join(path, 'u_real'), extent=[0, 1, -1, 1], x_label='t', y_label='x')
        show_image((u.numpy().reshape(512, 201) - u_pred.reshape(512, 201))**2, os.path.join(path, 'u_squared_error'), extent=[0, 1, -1, 1], format='%.1e', x_label='t', y_label='x')