import argparse
import tensorflow as tf

from train import train
from bayes_opt import BayesianOptimization

param_ranges = {
    'lr_0': (1, 10),
    'lr_1': (-6, -3),
    'T_0': (1, 10),
    'T_1': (-3, 2),
    'alpha': (0., 1.),
    'rho': (0., 1.)
}

def training_wrapper(meta_args):
    def inner_training_wrapper(lr_0, lr_1, T_0, T_1, alpha, rho) -> float:
        lr = lr_0 * 10**int(lr_1)
        T = T_0 * 10**int(T_1)
        print('arguments LR', lr, 'T', T, 'alpha', alpha, 'rho', rho)
        setattr(meta_args, 'layers', 3)
        setattr(meta_args, 'nodes', 256)
        setattr(meta_args, 'lr', lr)
        setattr(meta_args, 'T', T)
        setattr(meta_args, 'alpha', alpha)
        setattr(meta_args, 'rho', rho)

        loss = train(meta_args)
        tf.keras.backend.clear_session()
        return -loss

    xgbBO = BayesianOptimization(inner_training_wrapper, param_ranges)
    xgbBO.maximize(n_iter=80, init_points=15, acq='ei')
    print('BEST PARAMS', xgbBO.max['params'])


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='experiments', type=str, help='path where to store the results')

parser.add_argument('--network', default='fc', type=str, help='type of network')

parser.add_argument('--optimizer', default='adam', type=str, help='type of optimizer')
parser.add_argument('--patience', default=3, type=int, help='how many evaluations without improvement to wait before reducing learning rate')
parser.add_argument('--factor', default=.1, type=float, help='multiplicative factor by which to reduce the learning rate')

parser.add_argument('--task', default='helmholtz', type=str, help='type of task to fit')
parser.add_argument('--update_rule', default='relobalo', type=str, help='type of balancing')
parser.add_argument('--aggregate_boundaries', action='store_true', help='aggregate all boundary terms into one before balancing')

parser.add_argument('--epochs', default=100000, type=int, help='number of epochs')
parser.add_argument('--resample', action='store_true', help='resample datapoints or keep them fixed')
parser.add_argument('--batch_size', default=1024, type=int, help='number of sampled points in a batch')
parser.add_argument('--verbose', action='store_true', help='print progress to terminal')

args = parser.parse_args()
training_wrapper(args)
