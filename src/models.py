import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def fully_connected(nlayers, nnodes, activation='tanh', name='fully_connected'):
    xy = Input((2,))
    u = xy
    for i in range(nlayers):
        u = Dense(nnodes, activation=activation, name='dense'+str(i))(u)
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
        return [x*w for w, x in zip(self.W, X)]