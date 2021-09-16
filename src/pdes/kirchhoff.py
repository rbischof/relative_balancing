import numpy as np
import tensorflow as tf

TOL = 1e-5

class Kirchhoff():
    def __init__(self, E=30000, nue=.2, h=.2, a=10., b=10, p0=.015):
        self.E = E
        self.nue = nue
        self.h = h
        self.a = a
        self.b = b
        self.p0 = p0
        self.D = (E * h**3)/(12*(1-nue**2))
        self.num_b_losses = 12

    def training_batch(self, batch_size=1024):
        # sample internal area
        train_internal = np.random.uniform([0,0], [self.a, self.b], size=(batch_size//2, 2))
    
        # sample boundaries
        train_BC = [np.concatenate([tf.zeros((batch_size//8, 1)), 
                                    self.a*tf.ones((batch_size//8, 1)), 
                                    np.random.uniform(0, self.a, (batch_size//8, 1)), 
                                    np.random.uniform(0, self.a, (batch_size//8, 1))], axis=0), 
                    np.concatenate([np.random.uniform(0, self.b, (batch_size//8, 1)), 
                                    np.random.uniform(0, self.b, (batch_size//8, 1)), 
                                    tf.zeros((batch_size//8, 1)),
                                    self.b*tf.ones((batch_size//8, 1))], axis=0)]
                                
        x = tf.cast(tf.concat([train_BC[0], train_internal[:,:1]], axis=0), dtype=tf.float32)
        y = tf.cast(tf.concat([train_BC[1], train_internal[:,1:]], axis=0), dtype=tf.float32)
    
        return x, y

    def validation_batch(self):
        x, y = self.training_batch()
        w = tf.cast(self.p0 / (np.pi**4 * self.D * ((1/self.a**2 + 1/self.b**2)**2)) * \
            tf.math.sin(np.pi*x/self.a) * tf.math.sin(np.pi*y/self.b), dtype=tf.float32)
        return x, y, w

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
    def calculate_loss(self, model, x, y, aggregate_boundaries=False, training=False):
        W, _, _, Mx, My, dW_dxxxx, dW_dyyyy, dW_dxxyy = self.derivatives(model, x, y, training=training)
        
        # governing equation loss
        F = dW_dxxxx + 2*dW_dxxyy + dW_dyyyy
        f_loss = tf.reduce_mean((self.p0*tf.math.sin(np.pi*x/self.a)*tf.math.sin(np.pi*y/self.b) - F*self.D)**2)

        # boundary conditions loss
        xl = tf.cast(x < TOL, dtype=tf.float32)
        xu = tf.cast(x > self.a - TOL, dtype=tf.float32)
        yl = tf.cast(y < TOL, dtype=tf.float32)
        yu = tf.cast(y > self.b - TOL, dtype=tf.float32)

        if aggregate_boundaries:
            b_loss = tf.reduce_mean(((xl + xu + yl + yu)*W)**2 + \
                ((xl + xu)*Mx)**2 + ((yl + yu)*My)**2)
            return f_loss, [b_loss]
        else:
            b1_loss = tf.reduce_mean((xl*W)**2)
            b2_loss = tf.reduce_mean((xu*W)**2)
            b3_loss = tf.reduce_mean((yl*W)**2)
            b4_loss = tf.reduce_mean((yu*W)**2)
            b5_loss = tf.reduce_mean((xl*Mx)**2)
            b6_loss = tf.reduce_mean((xu*Mx)**2)
            b7_loss = tf.reduce_mean((yl*My)**2)
            b8_loss = tf.reduce_mean((yu*My)**2)
            return f_loss, [b1_loss, b2_loss, b3_loss, b4_loss, 
                            b5_loss, b6_loss, b7_loss, b8_loss]

    @tf.function
    def validation_loss(self, model, x, y, w):
        w_pred = model(tf.concat([x, y], axis=-1), training=False)
        return tf.reduce_mean((w - w_pred)**2)

