import os
import numpy as np
import tensorflow as tf

from utils import show_image
from scipy.interpolate import griddata
from tensorflow.experimental.numpy import isclose

TOL = 1e-5

class DiffusionSorption1D:
    def __init__(self, inverse:bool=False, inverse_var:float=None):
        super(DiffusionSorption1D, self).__init__()
        self.inverse = inverse
        self.num_b_losses = 3 if not inverse else 1
        self.D = inverse_var if inverse_var is not None else 5e-4

        self.Phi = 0.29      # porosity of the medium
        self.rho_S = 2880    #  bulk density
        self.kF = 3.5e-4     # Freundlich's parameter
        self.nF = 0.874      # Freundlich's exponent
        self.D = 5e-4        # effective diffusion coefficient 

        if not np.isclose(self.D, 5e-4):
            print('note: no validation data available for effective diffusion coefficient ', self.D, '; validation loss will be wrong.')
        self.data_min = (0, 0)
        self.data_max = (1, 500)

        self.u = np.load('data/diff_sorp/u.npy')
        self.x = np.load('data/diff_sorp/x.npy')
        self.t = np.load('data/diff_sorp/t.npy')
        self.ic_u = self.u[0].reshape(-1, 1)
        self.ic_x = self.x[0].reshape(-1, 1)
        self.ic_t = self.t[0].reshape(-1, 1)
        self.val_u = tf.cast(self.u[::1, ::10].reshape(-1, 1), dtype=tf.float32)
        self.val_x = tf.cast(self.x[::1, ::10].reshape(-1, 1), dtype=tf.float32)
        self.val_t = tf.cast(self.t[::1, ::10].reshape(-1, 1), dtype=tf.float32)
        self.u = self.u.reshape(-1, 1)
        self.x = self.x.reshape(-1, 1)
        self.t = self.t.reshape(-1, 1)

    def training_batch(self, batch_size:int=1024, incl_u:bool=False):
        internal = int(2 * batch_size // 3)
        initial = (batch_size - internal) // 2
        boundary_upper = (batch_size - internal - initial) // 2
        boundary_lower = batch_size - internal - initial - boundary_upper

        in_ixs = np.random.randint(len(self.u), size=internal)
        ic_ixs = np.random.randint(len(self.ic_u), size=initial)

        x = tf.cast(np.concatenate([self.x[in_ixs], 
                                    self.ic_x[ic_ixs], 
                                    np.ones((boundary_upper, 1)),
                                    np.zeros((boundary_lower, 1))], axis=0), dtype=tf.float32)
        t = tf.cast(np.concatenate([self.t[in_ixs], 
                                    self.ic_t[ic_ixs], 
                                    np.random.uniform(size=(boundary_upper, 1)),
                                    np.random.uniform(size=(boundary_lower, 1))], axis=0), dtype=tf.float32)
        if incl_u:
            u = tf.cast(np.concatenate([self.u[in_ixs], 
                                        self.ic_u[ic_ixs], 
                                        np.zeros((boundary_upper, 1)),
                                        np.ones((boundary_lower, 1))], axis=0), dtype=tf.float32)
            return x, t, u
        else:
            return x, t
    
    def validation_batch(self):
        return self.val_x, self.val_t, self.val_u

    def calculate_loss(self, model:tf.keras.Model, x, t, aggregate_boundaries:bool=False, training:bool=False):
        x, t, u = self.training_batch(incl_u=True)
        
        # predictions and derivatives
        u_pred = model[0]([x, t], training=training)
        du_dx, du_dt = tf.gradients(u_pred, [x, t])
        du_dxx = tf.gradients(du_dx, x)[0]

        # governing equation loss
        R = 1 + ((1-self.Phi)/self.Phi)*self.rho_S*self.kF*self.nF*(u_pred**(self.nF-1))

        if self.inverse:
            f_loss = tf.reduce_mean((du_dt - (model[1] / R)*du_dxx)**2)
            w_loss = tf.reduce_mean((u_pred - u)**2)
            return f_loss, [w_loss]
        else:
            f_loss = tf.reduce_mean((du_dt - (self.D / R)*du_dxx)**2)

            # boundary/initial conditions loss
            xl = tf.cast(isclose(x, 0, rtol=0., atol=TOL), dtype=tf.float32)
            xu = tf.cast(isclose(x, 1, rtol=0., atol=TOL), dtype=tf.float32)
            tl = tf.cast(isclose(t, 0, rtol=0., atol=TOL), dtype=tf.float32)

            bc_loss1 = tf.reduce_mean((u_pred - 1)**2 * xl)
            bc_loss2 = tf.reduce_sum((self.D * du_dx * xu)**2)
            ic_loss = tf.reduce_mean((u_pred - u)**2 * tl)
            b_losses = [ic_loss, bc_loss1, bc_loss2]

            if aggregate_boundaries:
                return f_loss, [tf.reduce_sum(b_losses)]
            else:
                return f_loss, [ic_loss, bc_loss1, bc_loss2]

    @tf.function
    def validation_loss(self, model:tf.keras.Model, x, t, u):
        u_pred = model[0]([x, t], training=False)
        if not self.inverse:
            return tf.reduce_mean((u - u_pred)**2)
        else:
            return tf.reduce_mean((u - u_pred)**2), tf.reduce_mean((model[1] - self.k)**2)

    def visualise(self, model:tf.keras.Model, path:str=None):
        x, t, u = self.validation_batch()
        u_pred = model[0].predict([x, t])

        show_image(np.rot90(u_pred.reshape(101, 103)), os.path.join(path, 'u_predicted'), extent=[0, 501, 0, 1], x_label='t', y_label='x')
        show_image(np.rot90(u.numpy().reshape(101, 103)), os.path.join(path, 'u_real'), extent=[0, 501, 0, 1], x_label='t', y_label='x')
        show_image(np.rot90(((u.numpy() - u_pred)**2).reshape(101, 103)), os.path.join(path, 'u_squared_error'), extent=[0, 501, 0, 1], format='%.2e', x_label='t', y_label='x')