import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Flatten, BatchNormalization, Dropout, Conv2DTranspose, Concatenate


def fully_connected(nlayers, nnodes, activation='tanh', name='fully_connected'):
    xy = Input((2,))
    u = xy
    for i in range(nlayers):
        u = Dense(nnodes, activation=activation, name='dense'+str(i))(u)
    u = Dense(1)(u)
    return Model(xy, u, name=name)

def partially_differentiable(nlayers, nnodes, name='partially_differentiable'):
    xy = Input((2,))
    u = xy
    for i in range(nlayers):
        u = Dense(3*nnodes, name='dense'+str(i))(u)
        u0 = tf.nn.relu(u[:, :nnodes])
        u1 = tf.nn.relu(100*u[:, nnodes:2*nnodes])*tf.nn.relu(u[:, nnodes:2*nnodes])
        u2 = tf.nn.relu(1000*u[:, 2*nnodes:3*nnodes])*tf.nn.relu(u[:, 2*nnodes:3*nnodes])*tf.nn.relu(u[:, 2*nnodes:3*nnodes])
        u = Concatenate()([u0, u1, u2])
    u = Dense(1)(u)
    return Model(xy, u, name=name)
    

class GradNormArgs(tf.Module):
    def __init__(self, nterms, alpha, **kwargs):
        super().__init__(**kwargs)
        self.W = [tf.Variable(1., trainable=True, name='w'+str(i)) for i in range(nterms)]
        self.L = [tf.constant(1.)]*nterms
        self.L_set = False
        self.alpha = alpha
    
    def set_L(self, l):
        if not self.L_set:
            self.L = [tf.constant(li) for li in l]
            self.L_set = True
    
    def __call__(self, X):
        W_n = tf.nn.softmax(self.W)
        return [X[i]*W_n[i] for i in range(len(X))]

def autoencoder():
    x = Input((28, 28), name='in')
    r = Reshape((28, 28, 1))(x)
    c = Conv2D(4, 3, padding='same', strides=2, activation='relu')(r)
    c = Conv2D(8, 3, padding='same', strides=2, activation='relu')(c)
    c = Conv2D(16, 3, padding='same', strides=2, activation='relu')(c)
    c = Conv2D(32, 3, padding='same', strides=2, activation='relu')(c)
    f = Flatten()(c)
    b = BatchNormalization()(f)
    dr = Dropout(.1)(b)
    d = Dense(128, activation='relu')(dr)
    r = Reshape((2, 2, 32))(d)
    u = Conv2DTranspose(32, 5, strides=2, activation='relu', padding='same')(r)
    c = Conv2D(16, 3, padding='same', activation='relu')(u)
    u = Conv2DTranspose(16, 5, strides=2, activation='relu', padding='same')(c)
    c = Conv2D(8, 3, padding='same', activation='relu')(u)
    u = Conv2DTranspose(8, 5, strides=2, activation='relu', padding='same')(c)
    c = Conv2D(4, 3, padding='valid', activation='relu')(u)
    u = Conv2DTranspose(4, 5, strides=2, activation='relu', padding='same')(c)
    c = Conv2D(4, 3, padding='same', activation='relu')(u)
    c = Conv2D(1, 3, padding='same')(c)

    model = tf.keras.Model(x, c, name='autoencoder')
    return model