import argparse

from train import train
from bayes_opt import BayesianOptimization

param_ranges = {
    'layers': (1, 6),
    'nodes': (16, 512),
    'lr_0': (1, 10),
    'lr_1': (-6, -3),
    'T_0': (1, 10),
    'T_1': (-3, 2),
    'alpha': (0.9, 0.9999999),
    'rho': (0, 1)
}

def training_wrapper(meta_args):
    def inner_training_wrapper(layers, nodes, lr_0, lr_1, T_0, T_1, alpha, rho) -> float:
        layers = int(layers)
        nodes = int(nodes)
        lr = lr_0 * 10**int(lr_1)
        T = T_0 * 10**int(T_1)

        setattr(meta_args, 'layers', layers)
        setattr(meta_args, 'nodes', nodes)
        setattr(meta_args, 'lr', lr)
        setattr(meta_args, 'T', T)
        setattr(meta_args, 'alpha', alpha)
        setattr(meta_args, 'rho', rho)

        return -train(meta_args)

    xgbBO = BayesianOptimization(inner_training_wrapper, param_ranges)
    xgbBO.maximize(n_iter=100, init_points=20, acq='ei')


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='experiments/', type=str, help='path where to store the results')

parser.add_argument('--network', default='fc', type=str, help='type of network')

parser.add_argument('--optimizer', default='adam', type=str, help='type of optimizer')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--patience', default=3, type=int, help='how many evaluations without improvement to wait before reducing learning rate')
parser.add_argument('--factor', default=.1, type=float, help='multiplicative factor by which to reduce the learning rate')

parser.add_argument('--pde', default='helmholtz', type=str, help='type of pde to fit')
parser.add_argument('--update_rule', default='manual', type=str, help='type of balancing')
parser.add_argument('--aggregate_boundaries', action='store_true', help='aggregate all boundary terms into one before balancing')

parser.add_argument('--epochs', default=100000, type=int, help='number of epochs')
parser.add_argument('--resample', action='store_true', help='resample datapoints or keep them fixed')
parser.add_argument('--batch_size', default=1024, type=int, help='number of sampled points in a batch')
parser.add_argument('--verbose', action='store_true', help='print progress to terminal')

args = parser.parse_args()
training_wrapper(args)