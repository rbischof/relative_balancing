from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def fully_connected(nlayers, nnodes, activation='tanh', name='fully_connected'):
    return Sequential([Dense(nnodes, activation=activation) for _ in range(nlayers)] + [Dense(1)], name=name)