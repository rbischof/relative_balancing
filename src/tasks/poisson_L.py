import os
import numpy as np
import tensorflow as tf

from utils import show_image
from scipy.interpolate import griddata
from tensorflow.experimental.numpy import isclose

TOL = 1e-5

class Poisson_L():
    def __init__(self, inverse:bool=False, inverse_var:float=None):
        self.inverse = inverse
        self.num_b_losses = 6 if not inverse else 1
        self.k = inverse_var if inverse_var is not None else 1

        data = np.load('data/Poisson_Lshape.npz', allow_pickle=True)
        self.val_u = data['y_ref'][::4]
        self.val_x = tf.cast(data['X_test'][::4, :1][np.isnan(self.val_u[:, 0]) != True], dtype=tf.float32)
        self.val_y = tf.cast(data['X_test'][::4, 1:][np.isnan(self.val_u[:, 0]) != True], dtype=tf.float32)
        self.val_u = tf.cast(self.val_u[np.isnan(self.val_u[:, 0]) != True], dtype=tf.float32)

    def training_batch(self, batch_size:int=1024):
        internal = 2 * (batch_size // 3)
        boundary = batch_size - internal

        # sample internal area
        internal1 = np.random.uniform(low=[-0.1, -1.1], high=[1.1, 0.1], size=(internal//3, 2)) # sample small square for L
        internal2 = np.random.uniform(low=[-1.1, -1.1], high=[0.1, 1.1], size=(internal - internal//3, 2)) # sample rectangle
        
        # sample boundries
        BCx1 = -np.ones((boundary//4, 1))
        BCy1 = np.random.uniform(-1, 1, (boundary//4, 1))
        BCx2 = np.random.uniform(-1, 1, (boundary//4, 1))
        BCy2 = -np.ones((boundary//4, 1))
        BCx3 = np.concatenate([np.ones((boundary//8, 1)), np.zeros((boundary//8, 1))], axis=0)
        BCy3 = np.concatenate([np.random.uniform(-1, 0, ((boundary//8), 1)), np.random.uniform(0, 1, ((boundary//8), 1))], axis=0)
        last_len0 = (boundary - (len(BCx1) + len(BCx2) + len(BCx3))) // 2
        last_len1 = last_len0 + (boundary - (len(BCx1) + len(BCx2) + len(BCx3))) % 2
        BCx4 = np.concatenate([np.random.uniform(-1, 0, (last_len0, 1)), np.random.uniform(0, 1, (last_len1, 1))], axis=0)
        BCy4 = np.concatenate([np.ones((last_len0, 1)), np.zeros((last_len1, 1))], axis=0)

        x = tf.constant(np.concatenate([internal1[:,:1], internal2[:,:1], BCx1, BCx2, BCx3, BCx4], axis=0), dtype=tf.float32)
        y = tf.constant(np.concatenate([internal1[:,1:], internal2[:,1:], BCy1, BCy2, BCy3, BCy4], axis=0), dtype=tf.float32)

        return x, y

    def validation_batch(self):
        return self.val_x, self.val_y, self.val_u

    def calculate_loss(self, model:tf.keras.Model, x, y, aggregate_boundaries:bool=False, training:bool=False):
        if self.inverse:
            x, y, u = self.validation_batch()
        # predictions and derivatives
        u_pred = model[0](tf.concat([x, y], axis=-1), training=training)
        du_dx, du_dy = tf.gradients(u_pred, [x, y])
        du_dxx = tf.gradients(du_dx, x)[0]
        du_dyy = tf.gradients(du_dy, y)[0]
        f = du_dxx + du_dyy

        if self.inverse:
            f_loss = tf.reduce_mean((f + model[1])**2)
            u_loss = tf.reduce_mean((u_pred - u)**2)
            return f_loss, [u_loss]
        else:
            f_loss = tf.reduce_mean((f + self.k)**2)

            # boundary conditions loss
            xl = tf.cast(isclose(x, -1, rtol=0., atol=TOL), dtype=tf.float32)
            xu0 = tf.cast(tf.math.logical_and(isclose(x, 0, rtol=0., atol=TOL), y >= 0), dtype=tf.float32)
            xu1 = tf.cast(tf.math.logical_and(isclose(x, 1, rtol=0., atol=TOL), y <= 0), dtype=tf.float32)
            yl = tf.cast(isclose(y, -1, rtol=0., atol=TOL), dtype=tf.float32)
            yu0 = tf.cast(tf.math.logical_and(isclose(y, 1, rtol=0., atol=TOL), x <= 0), dtype=tf.float32)
            yu1 = tf.cast(tf.math.logical_and(isclose(y, 0, rtol=0., atol=TOL), x >= 0), dtype=tf.float32)

            if aggregate_boundaries:
                b_loss = tf.reduce_mean((u_pred * (xl + xu0 + xu1 + yl + yu0 + yu1))**2)
                return f_loss, [b_loss]
            else:
                b1_loss = 10*tf.reduce_mean((u_pred * xl)**2)
                b2_loss = 100*tf.reduce_mean((u_pred * xu0)**2)
                b3_loss = 10*tf.reduce_mean((u_pred * xu1)**2)
                b4_loss = 10*tf.reduce_mean((u_pred * yl)**2)
                b5_loss = 10*tf.reduce_mean((u_pred * yu0)**2)
                b6_loss = 100*tf.reduce_mean((u_pred * yu1)**2)
                return f_loss, [b1_loss, b2_loss, b3_loss, b4_loss, b5_loss, b6_loss]

    @tf.function
    def validation_loss(self, model:tf.keras.Model, x, y, u):
        u_pred = model[0](tf.concat([x, y], axis=-1), training=False)
        if not self.inverse:
            return tf.reduce_mean((u - u_pred)**2)
        else:
            return tf.reduce_mean((u - u_pred)**2), tf.reduce_mean((model[1] - self.k)**2)

    def L_coord_to_img(self, x, y, u):
        img_width = 64
        x, y, u = [i.numpy().flatten() if not isinstance(i, np.ndarray) else i.flatten() for i in [x, y, u]]
        grid_x, grid_y = np.mgrid[-1:1:complex(0, img_width), -1:1:complex(0, img_width)]
        img = griddata((x, y), u, (grid_x, grid_y), fill_value=0., method='linear')
        img[img_width//2:, img_width//2:] = 0
        return img

    def visualise(self, model:tf.keras.Model, path:str=None):
        x, y, u = self.validation_batch()
        u_pred = model[0].predict(tf.concat([x, y], axis=-1))

        show_image(self.L_coord_to_img(x, y, u_pred), os.path.join(path, 'u_predicted'), extent=[-1, 1, -1, 1])
        show_image(self.L_coord_to_img(x, y, u), os.path.join(path, 'u_real'), extent=[-1, 1, -1, 1])
        show_image(self.L_coord_to_img(x, y, ((u.numpy() - u_pred)**2)), os.path.join(path, 'u_squared_error'), extent=[-1, 1, -1, 1], format='%.2e')