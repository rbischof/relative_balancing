import numpy as np
import tensorflow as tf

TOL = 1e-5

class Kirchhoff():
    def __init__(self, batch_size=1024, batches_per_epoch=1000, E=30000, nue=.2, h=.2, a=10., b=10, p0=.015):
        self.generator = KirchhoffDataGenerator(batch_size, batches_per_epoch, a, b, p0)
        self.num_b_losses = 12
        self.E = E
        self.nue = nue
        self.h = h
        self.a = a
        self.b = b
        self.p0 = p0
        self.D = (E * h**3)/(12*(1-nue**2))

    def generate_data(self):
        return self.generator.__getitem__(None)

    @tf.function
    def derivatives(self, model, x, y, training=False):
        W = model(tf.concat([x, y], axis=-1), training=training)
        dW_dx, dW_dy = tf.gradients(W, [x, y])
        dW_dxx = tf.gradients(dW_dx, x)[0]
        dW_dyy = tf.gradients(dW_dy, y)[0]
        dW_dxxx, dW_dxxy = tf.gradients(dW_dxx, [x, y])
        dW_dyyy = tf.gradients(dW_dyy, y)[0]
        dW_dxxxx = tf.gradients(dW_dxxx, x)[0]
        dW_dxxyy = tf.gradients(dW_dxxy, y)[0]
        dW_dyyyy = tf.gradients(dW_dyyy, y)[0]
        Mx = -self.D*(dW_dxx + self.nue*dW_dyy)
        My = -self.D*(self.nue*dW_dxx + dW_dyy)
        return W, dW_dx, dW_dy, Mx, My, dW_dxxxx, dW_dyyyy, dW_dxxyy

    @tf.function
    def calculate_loss(self, model, x, y, w, aggregate_boundaries=False, training=False):
        W, dW_dx, dW_dy, Mx, My, dW_dxxxx, dW_dyyyy, dW_dxxyy = self.derivatives(model, x, y, training=training)
        
        # governing equation loss
        F = dW_dxxxx + 2*dW_dxxyy + dW_dyyyy
        f_loss = tf.reduce_mean((self.p0*tf.math.sin(np.pi*x/self.a)*tf.math.sin(np.pi*y/self.b) - F*self.D)**2)

        # boundary conditions loss
        xl = tf.cast(x < TOL, dtype=tf.float32)
        xu = tf.cast(x > self.a - TOL, dtype=tf.float32)
        yl = tf.cast(y < TOL, dtype=tf.float32)
        yu = tf.cast(y > self.b - TOL, dtype=tf.float32)

        val_loss = tf.reduce_mean((w - W)**2)

        if aggregate_boundaries:
            b_loss = tf.reduce_mean(((xl + xu + yl + yu)*W)**2 + \
                ((xl + xu)*dW_dy)**2 + ((yl + yu)*dW_dx)**2 + \
                ((xl + xu)*Mx)**2 + ((yl + yu)*My)**2)
            return f_loss, [b_loss], val_loss
        else:
            b1_loss  = tf.reduce_mean((xl*W)**2)
            b2_loss  = tf.reduce_mean((xu*W)**2)
            b3_loss  = tf.reduce_mean((yl*W)**2)
            b4_loss  = tf.reduce_mean((yu*W)**2)
            b5_loss  = tf.reduce_mean((xl*dW_dy)**2)
            b6_loss  = tf.reduce_mean((xu*dW_dy)**2)
            b7_loss  = tf.reduce_mean((yl*dW_dx)**2)
            b8_loss  = tf.reduce_mean((yu*dW_dx)**2)
            b9_loss  = tf.reduce_mean((xl*Mx)**2)
            b10_loss = tf.reduce_mean((xu*Mx)**2)
            b11_loss = tf.reduce_mean((yl*My)**2)
            b12_loss = tf.reduce_mean((yu*My)**2)
            return f_loss, [b1_loss, b2_loss, b3_loss, b4_loss, b5_loss, b6_loss,
                            b7_loss, b8_loss, b9_loss, b10_loss, b11_loss, b12_loss], val_loss

class KirchhoffDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, batches_per_epoch=1000, a=10., b=10., p0=.015):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.a = a
        self.b = b
        self.p0 = p0

    def __len__(self):
        return self.batches_per_epoch
    
    def __data_generation(self, list_IDs_temp):
        # sample internal area
        train_internal = np.random.uniform(low=[0,0], high=[self.a, self.b], size=(self.batch_size//2, 2))
    
        # sample boundries
        train_BC = [np.concatenate([np.zeros((self.batch_size//8, 1)), 
                                    self.a*np.ones((self.batch_size//8, 1)), 
                                    np.random.uniform(0, self.a, (self.batch_size//8, 1)), 
                                    np.random.uniform(0, self.a, (self.batch_size//8, 1))], axis=0), 
                    np.concatenate([np.random.uniform(0, self.b, (self.batch_size//8, 1)), 
                                    np.random.uniform(0, self.b, (self.batch_size//8, 1)), 
                                    np.zeros((self.batch_size//8, 1)),
                                    self.b*np.ones((self.batch_size//8, 1))], axis=0)]
                                
        x = tf.cast(tf.concat([train_BC[0], train_internal[:,:1]], axis=0), dtype=tf.float32)
        y = tf.cast(tf.concat([train_BC[1], train_internal[:,1:]], axis=0), dtype=tf.float32)
        w = self.p0 / (np.pi**4 * ((1/self.a**2 + 1/self.b**2)**2)) * \
            tf.math.sin(np.pi*x/self.a) * tf.math.sin(np.pi*y*self.b)
    
        return x, y, w

    def __getitem__(self, index):
        'Generate one batch of data'
        return self.__data_generation(None)