import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Flatten, BatchNormalization, Dropout, Conv2DTranspose, Concatenate

class PINNGlorotNormal(tf.keras.initializers.Initializer):
  def __call__(self, shape, dtype=None, **kwargs):
    return (tf.cast(tf.experimental.numpy.random.randint(0, 2, shape, dtype=tf.experimental.numpy.int32) * 2 - 1, dtype=tf.float32) + tf.keras.initializers.GlorotNormal().__call__(shape, dtype, **kwargs)) / shape[0]


def fully_connected(nlayers, nnodes, data_min:tuple, data_max:tuple, activation=tf.math.sin, name='fully_connected'):
    x, y = Input((1,), name='x'), Input((1,), name='y')

    u = Concatenate()([(x - data_min[0]) / (data_max[0] - data_min[0]), (y - data_min[1]) / (data_max[1] - data_min[1])]) * 2 - 1
    u = Dense(nnodes, activation=activation, kernel_initializer=tf.keras.initializers.GlorotNormal(), name='dense0')(u)
    for i in range(1, nlayers):
        u = Dense(nnodes, activation=activation, kernel_initializer=tf.keras.initializers.GlorotNormal(), name='dense'+str(i))(u) + u
    u = Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotNormal())(u)
    return Model([x, y], u, name=name)
    

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
