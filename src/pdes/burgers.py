import numpy as np
import tensorflow as tf

TOL = 1e-5

class Burgers():
    def __init__(self, batch_size=1024, batches_per_epoch=1000):
        self.generator = BurgersDataGenerator(batch_size, batches_per_epoch)
        self.num_b_losses = 3

    def generate_data(self):
        return self.generator.__getitem__(None)

    def calculate_loss(self, model, x, t, u, aggregate_boundaries=False, training=False):
        # predictions and derivatives
        u_pred = model(tf.concat([x, t], axis=-1), training=False)
        du_dx, du_dt = tf.gradients(u_pred, [x, t])
        du_dxx = tf.gradients(du_dx, x)[0]
        f_pred = du_dt + u_pred*du_dx - (0.01/np.pi)*du_dxx

        # governing equation loss
        f_loss = tf.reduce_mean(f_pred**2)

        # boundary conditions loss
        xl = tf.cast(x < (-1 + TOL), dtype=tf.float32)
        xu = tf.cast(x > ( 1 - TOL), dtype=tf.float32)
        tl = tf.cast(t < TOL, dtype=tf.float32)
        b1_loss = tf.reduce_mean((u_pred * xu)**2)
        b2_loss = tf.reduce_mean((u_pred * xl)**2)
        b3_loss = tf.reduce_mean(((-tf.math.sin(np.pi*x) - u_pred) * tl)**2)

        val_loss = tf.reduce_mean((u - u_pred)**2)

        if aggregate_boundaries:
            b_loss = tf.reduce_mean((u_pred * (xl + xu + tl))**2)
            return f_loss, [b_loss], val_loss
        else:
            b1_loss = tf.reduce_mean((u_pred * xl)**2)
            b2_loss = tf.reduce_mean((u_pred * xu)**2)
            b3_loss = tf.reduce_mean((u_pred * tl)**2)
            return f_loss, [b1_loss, b2_loss, b3_loss], val_loss


class BurgersDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, batches_per_epoch=1000):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        return self.batches_per_epoch
    
    def __data_generation(self, list_IDs_temp):
        x_in = tf.random.uniform((2*self.batch_size//3, 1), minval=-1, maxval=1, dtype=tf.float32)
        x_b1 = tf.random.uniform((self.batch_size//9, 1), minval=-1, maxval=(-1+TOL), dtype=tf.float32)
        x_b2 = tf.random.uniform((self.batch_size//9, 1), minval=(1-TOL), maxval=1, dtype=tf.float32)
        x_b3 = tf.random.uniform((self.batch_size//9, 1), minval=-1, maxval=1, dtype=tf.float32)
        x = tf.concat([x_in, x_b1, x_b2, x_b3], axis=0)

        t_in = tf.random.uniform((2*self.batch_size//3, 1), minval=0, maxval=1, dtype=tf.float32)
        t_b1 = tf.random.uniform((self.batch_size//9, 1), minval=0, maxval=1, dtype=tf.float32)
        t_b2 = tf.random.uniform((self.batch_size//9, 1), minval=0, maxval=1, dtype=tf.float32)
        t_b3 = tf.random.uniform((self.batch_size//9, 1), minval=0, maxval=TOL, dtype=tf.float32)
        t = tf.concat([t_in, t_b1, t_b2, t_b3], axis=0)

        u = tf.math.sin(np.pi*x)*(t - 1)
        return x, t, u

    def __getitem__(self, index):
        'Generate one batch of data'
        return self.__data_generation(None)