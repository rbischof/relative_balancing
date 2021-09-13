import os
import numpy as np
import tensorflow as tf

import argparse
from time import time, strftime, gmtime

from models import fully_connected, GradNormArgs
from update_rules import manual, lrannealing, softadapt, softadapt_rnd_lookback, gradnorm
from pdes.helmholtz import Helmholtz
from pdes.burgers import Burgers
from pdes.kirchhoff import Kirchhoff
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

def train(meta_args):

    # initialize network
    if meta_args.network == 'fc':   
        model = fully_connected(meta_args.layers, meta_args.nodes)
    else:
        raise ValueError('Network type not understood:' + meta_args.network)

    # initialize optimizer
    if meta_args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=meta_args.lr)
    elif meta_args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=meta_args.lr)
    else:
        raise ValueError('Optimizer type not understood:' + meta_args.optimizer)

    # initialize pde
    if meta_args.pde == 'helmholtz':
        pde = Helmholtz()
    elif meta_args.pde == 'burgers':
        pde = Burgers()
    elif meta_args.pde == 'kirchhoff':
        pde = Kirchhoff()
    else:
        raise ValueError('PDE type not understood:' + meta_args.pde)

    num_b_losses = 1 if meta_args.aggregate_boundaries else pde.num_b_losses

    # initialize update rule
    if meta_args.update_rule == 'manual':
        update_rule = manual
        args = {"lam"+str(i): tf.constant(1.) for i in range(num_b_losses+1)}
        alpha = [1.]
        args.update({"alpha": tf.constant(alpha[0], dtype=tf.float32)})
    elif meta_args.update_rule == 'lrannealing':
        update_rule = lrannealing
        args = {"lam"+str(i): tf.constant(1.) for i in range(num_b_losses)}
        alpha = [[tf.constant(1.) for _ in range(99)] + [tf.constant(meta_args.alpha, dtype=tf.float32)]]*((meta_args.epochs+1)//100)
        alpha = [a for sub_alpha in alpha for a in sub_alpha]
        args.update({"alpha": tf.constant(alpha[0], dtype=tf.float32)})
    elif meta_args.update_rule == 'softadapt':
        update_rule = softadapt
        args = {"lam"+str(i): tf.constant(1.) for i in range(num_b_losses+1)}
        args.update({"l"+str(i): tf.constant(1.) for i in range(num_b_losses+1)})
        args.update({"T": tf.constant(meta_args.T, dtype=tf.float32)})
        alpha = [tf.constant(meta_args.alpha, dtype=tf.float32)]
        args.update({"alpha": tf.constant(alpha[0])})
    elif meta_args.update_rule == 'softadapt_rnd_lookback':
        update_rule = softadapt_rnd_lookback
        args = {"lam"+str(i): tf.constant(1.) for i in range(num_b_losses+1)}
        args.update({"l"+str(i): tf.constant(1.) for i in range(num_b_losses+1)})
        args.update({"l0"+str(i): tf.constant(1.) for i in range(num_b_losses+1)})
        args.update({"T": tf.constant(meta_args.T, dtype=tf.float32)})
        rho = (np.random.uniform(size=meta_args.epochs+1) > meta_args.rho).astype(int).astype(np.float32)
        args.update({'rho': tf.constant(rho[0], dtype=tf.float32)})
        alpha = [tf.constant(meta_args.alpha, tf.float32)]
        args.update({"alpha": alpha[0]})
    elif meta_args.update_rule == 'gradnorm':
        update_rule = gradnorm
        args = {"l"+str(i): tf.constant(1.) for i in range(num_b_losses+1)}
        alpha = [tf.constant(0.)]+[tf.constant(meta_args.T, dtype=tf.float32)]*(meta_args.epochs+1)
        args.update({"alpha": tf.constant(alpha[0], dtype=tf.float32)})
        model = [model, GradNormArgs(nterms=num_b_losses+1, alpha=alpha)]
        optimizer = [optimizer]*2
    else:
        raise ValueError('Update rule not understood:' + meta_args.update_rule)

    experiment_path = create_directory(os.path.join('experiments', meta_args.path))

    summary = []
    args_summary = []
    if isinstance(args, dict):
        args_keys = args.keys()
    best_loss = 1e9
    best_val_loss = 1e9
    cooldown = meta_args.patience
    meltdown = 0
    
    try:
        best_model = tf.keras.models.clone_model(model)
    except:
        best_model = model
        print("cloning model failed, saving best model separately is not possible")

    x, y = pde.training_batch(meta_args.batch_size)
    print('start training of', meta_args.pde, 'in', experiment_path)
    start = time()
    for epoch in range(meta_args.epochs):
        
        grads, f_loss, b_losses, oargs = update_rule(model, 
                            optimizer, 
                            pde, 
                            tf.constant(x, dtype=tf.float32), 
                            tf.constant(y, dtype=tf.float32), 
                            args,
                            meta_args.aggregate_boundaries)
        if isinstance(grads, tuple):
            for i, g in enumerate(grads):
                optimizer[i].apply_gradients(zip(g, model[i].trainable_variables))
        else:
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss = f_loss + tf.reduce_sum(b_losses)            
        
        if not meta_args.update_rule == 'gradnorm' or epoch == 0:
            args = oargs.copy()
        if (meta_args.update_rule == 'gradnorm' or meta_args.update_rule == 'softadapt_rnd_lookback') and epoch == 0:
            for i in range(num_b_losses+1):
                args['l0'+str(i)] = ([f_loss]+b_losses)[i]
        if meta_args.resample:
            x, y = pde.training_batch(meta_args.batch_size)
        if len(alpha) > 1:
            args['alpha'] = tf.constant(alpha[1], dtype=tf.float32)
            alpha = alpha[1:]
            args['rho'] = tf.constant(rho[1], dtype=tf.float32)
            rho = rho[1:]
                
        if epoch % 1000 == 0:
            x, y, u = pde.validation_batch()
            if isinstance(model, list):
                val_loss = pde.validation_loss(model[0], x, y, u)
            else:
                val_loss = pde.validation_loss(model, x, y, u)

            loss, f_loss, val_loss = gpu_to_numpy(reduce_mean_all([loss, f_loss, val_loss]))
            b_losses  = gpu_to_numpy(b_losses)

            summary.append([loss, val_loss, f_loss]+b_losses)
            
            if isinstance(args, dict):
                e_args = gpu_to_numpy(args.values())
                args_summary.append(e_args)
            else:
                e_args = gpu_to_numpy(args.variables())
                args_summary.append(e_args)
        
            # reduce lr or stop early if model doesn't improve after warmup phase
            if loss < best_loss:
                best_loss = loss
                best_val_loss = val_loss
                cooldown = meta_args.patience
                meltdown = 0
                if not isinstance(model, list):
                    best_model.set_weights(model.get_weights())
            if cooldown <= 0 and epoch > meta_args.epochs/10:
                cooldown = meta_args.patience
                meltdown += 1
                if not isinstance(model, list):
                    if meta_args.verbose:
                        print(' reducing LR to', (optimizer.lr*meta_args.factor).numpy())
                    tf.keras.backend.set_value(optimizer.lr, optimizer.lr*meta_args.factor)
                    model.set_weights(best_model.get_weights())
                else:
                    if meta_args.verbose:
                        print(' reducing LR to', (optimizer[0].lr*meta_args.factor).numpy())
                    tf.keras.backend.set_value(optimizer[0].lr, optimizer[0].lr*meta_args.factor)
                    tf.keras.backend.set_value(optimizer[1].lr, optimizer[1].lr*meta_args.factor)
            if meltdown > 4 or time()-start > 13500:
                if meta_args.verbose:
                    print('early stopping')
                break
            cooldown -= 1
    
            # print evaluation metrics
            if meta_args.verbose:
                print("epoch {0:<4}".format(epoch),
                      "loss {:<.3e}".format(best_loss), 
                      "val_loss {:<.3e}".format(val_loss),
                    "F {0:<.2e}".format(f_loss), 
                    "B", b_losses,
                    "ARGS" if not isinstance(args, dict) else list(zip(args_keys, e_args)),
                    strftime('%H:%M:%S', gmtime(time()-start)))

    append_to_results((time()-start)/epoch*1000, meta_args, best_loss, best_val_loss)
    # save results
    np.save(experiment_path+'/summary', summary)
    if args is not None:
        np.save(experiment_path+'/args_summary', args_summary)
    if not isinstance(model, list):
        best_model.save(experiment_path+'/model')
    else:
        model[0].save(experiment_path+'/model')

    return best_val_loss
        

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='experiments/', type=str, help='path where to store the results')

parser.add_argument('--layers', default=1, type=int, help='number of layers')
parser.add_argument('--nodes', default=32, type=int, help='number of nodes')
parser.add_argument('--network', default='fc', type=str, help='type of network')

parser.add_argument('--optimizer', default='adam', type=str, help='type of optimizer')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--patience', default=3, type=int, help='how many evaluations without improvement to wait before reducing learning rate')
parser.add_argument('--factor', default=.1, type=float, help='multiplicative factor by which to reduce the learning rate')

parser.add_argument('--pde', default='helmholtz', type=str, help='type of pde to fit')
parser.add_argument('--update_rule', default='manual', type=str, help='type of balancing')
parser.add_argument('--T', default=1., type=float, help='temperature parameter for softmax')
parser.add_argument('--alpha', default=.999, type=float, help='rate for exponential decay')
parser.add_argument('--rho', default=1., type=float, help='rate for exponential decay')
parser.add_argument('--aggregate_boundaries', action='store_true', help='aggregate all boundary terms into one before balancing')

parser.add_argument('--epochs', default=100000, type=int, help='number of epochs')
parser.add_argument('--resample', action='store_true', help='resample datapoints or keep them fixed')
parser.add_argument('--batch_size', default=1024, type=int, help='number of sampled points in a batch')
parser.add_argument('--verbose', action='store_true', help='print progress to terminal')

args = parser.parse_args()
train(args)
