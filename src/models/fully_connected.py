from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

class FullyConnected():
    def __init__(self, input_dim, output_dim, nlayers, nnodes, activation='tanh', name=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.nnodes = nnodes
        self.activation = activation
        self.name = name

        self.model = Sequential([Dense(self.n_nodes, activation=self.activation) for _ in range(self.nlayers)] + \
            [Dense(1)], name=self.name)