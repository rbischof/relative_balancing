from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def fully_connected(nlayers, nnodes, activation='tanh', name='fully_connected'):
    xy = Input((2,))
    u = xy
    for _ in range(nlayers):
        u = Dense(nnodes, activation=activation)(u)
    u = Dense(1)(u)
    return Model(xy, u, name=name)