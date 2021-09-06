import scipy.io
import numpy as np
import tensorflow as tf

TOL = 1e-5

class Burgers():
    def __init__(self):
        self.num_b_losses = 3

    def training_batch(self, batch_size=1024):
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

        return x, t
    
    def validation_batch(self):
        data = scipy.io.loadmat('validation_data/burgers_shock_mu_01_pi.mat')  	# Load data from file
        x = data['x']                                                        # 256 points between -1 and 1 [256x1]
        t = data['t']                                                        # 100 time points between 0 and 1 [100x1] 
        u = tf.cast(np.rot90(data['usol'], k=1)[::-1], dtype=tf.float32)     # solution of 256x100 grid points
        x, t = np.meshgrid(x,t)                                              # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
        return tf.cast(tf.reshape(x, (-1, 1)), dtype=tf.float32), tf.cast(tf.reshape(t, (-1, 1)), dtype=tf.float32), tf.reshape(u, (-1, 1))


    @tf.function
    def calculate_loss(self, model, x, t, aggregate_boundaries=False, training=False):
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

        if aggregate_boundaries:
            b_loss = tf.reduce_mean((u_pred * (xl + xu + tl))**2)
            return f_loss, [b_loss]
        else:
            b1_loss = tf.reduce_mean((u_pred * xu)**2)
            b2_loss = tf.reduce_mean((u_pred * xl)**2)
            b3_loss = tf.reduce_mean(((-tf.math.sin(np.pi*x) - u_pred) * tl)**2)
            return f_loss, [b1_loss, b2_loss, b3_loss]
    
    @tf.function
    def validation_loss(self, model, x, y, u):
        u_pred = model(tf.concat([x, y], axis=-1), training=False)
        return tf.reduce_mean((u - u_pred)**2)