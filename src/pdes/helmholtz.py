import numpy as np
import tensorflow as tf

TOL = 1e-5

class Helmholtz():
    def __init__(self, batch_size=1024, batches_per_epoch=1000):
        self.generator = HelmholtzDataGenerator(batch_size, batches_per_epoch)
        self.num_b_losses = 4

    def generate_data(self):
        return self.generator.__getitem__(None)

    def calculate_loss(self, model, x, y, u, aggregate_boundaries=False, training=False):
        print('helmholz', x, y)
        print(tf.concat([x, y], axis=-1))
        # predictions and derivatives
        u_pred = model(tf.concat([x, y], axis=-1), training=training)
        du_dx, du_dy = tf.gradients(u_pred, [x, y])
        du_dxx = tf.gradients(du_dx, x)[0]
        du_dyy = tf.gradients(du_dy, y)[0]
        f_pred = du_dxx + du_dyy + 1

        # governing equation loss
        sin_xy = tf.math.sin(np.pi*x)*tf.math.sin(4*np.pi*y)
        f_loss = tf.reduce_mean((f_pred + np.pi**2*sin_xy + (4*np.pi)**2*sin_xy - sin_xy)**2)

        # boundary conditions loss
        xl = tf.cast(x < (-1 + TOL), dtype=tf.float32)
        xu = tf.cast(x > ( 1 - TOL), dtype=tf.float32)
        yl = tf.cast(y < (-1 + TOL), dtype=tf.float32)
        yu = tf.cast(y > ( 1 - TOL), dtype=tf.float32)

        val_loss = tf.reduce_mean((u - u_pred)**2)

        if aggregate_boundaries:
            b_loss = tf.reduce_mean((u_pred * (xl + xu + yl + yu))**2)
            return f_loss, [b_loss]
        else:
            b1_loss = tf.reduce_mean((u_pred * xl)**2)
            b2_loss = tf.reduce_mean((u_pred * xu)**2)
            b3_loss = tf.reduce_mean((u_pred * yl)**2)
            b4_loss = tf.reduce_mean((u_pred * yu)**2)
            return f_loss, [b1_loss, b2_loss, b3_loss, b4_loss], val_loss


class HelmholtzDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, batches_per_epoch=1000):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        return self.batches_per_epoch
    
    def __data_generation(self, list_IDs_temp):
        x_in = tf.random.uniform((2*self.batch_size//3, 1), minval=-1, maxval=1, dtype=tf.float32)
        x_b1 = tf.random.uniform((self.batch_size//12, 1), minval=-1, maxval=(-1+TOL), dtype=tf.float32)
        x_b2 = tf.random.uniform((self.batch_size//12, 1), minval=(1-TOL), maxval=1, dtype=tf.float32)
        x_b3 = tf.random.uniform((self.batch_size//12, 1), minval=-1, maxval=1, dtype=tf.float32)
        x_b4 = tf.random.uniform((self.batch_size//12, 1), minval=-1, maxval=1, dtype=tf.float32)
        x = tf.concat([x_in, x_b1, x_b2, x_b3, x_b4], axis=0)

        y_in = tf.random.uniform((2*self.batch_size//3, 1), minval=-1, maxval=1, dtype=tf.float32)
        y_b1 = tf.random.uniform((self.batch_size//12, 1), minval=-1, maxval=1, dtype=tf.float32)
        y_b2 = tf.random.uniform((self.batch_size//12, 1), minval=-1, maxval=1, dtype=tf.float32)
        y_b3 = tf.random.uniform((self.batch_size//12, 1), minval=-1, maxval=(-1+TOL), dtype=tf.float32)
        y_b4 = tf.random.uniform((self.batch_size//12, 1), minval=(1-TOL), maxval=1, dtype=tf.float32)
        y = tf.concat([y_in, y_b1, y_b2, y_b3, y_b4], axis=0)

        u = tf.math.sin(np.pi*x)*tf.math.sin(4*np.pi*y)
        return x, y, u

    def __getitem__(self, index):
        'Generate one batch of data'
        return self.__data_generation(None)