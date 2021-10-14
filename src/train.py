import os
import argparse
import numpy as np
import tensorflow as tf

from time import time, strftime, gmtime

from tasks.mnist import MNIST
from tasks.burgers import Burgers
from tasks.helmholtz import Helmholtz
from tasks.kirchhoff import Kirchhoff
from models import fully_connected, GradNormArgs, autoencoder
from update_rules import manual, lrannealing, softadapt, relobalo, gradnorm
from utils import gpu_to_numpy, reduce_mean_all, append_to_results, create_directory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

def train(meta_args):

    # initialize network
    if meta_args.task == 'mnist':
        model = [autoencoder()]
        if meta_args.inverse:
            model += [tf.Variable(.5, trainable=True, name='inverse_var')]
    elif meta_args.network == 'fc':   
        model = [fully_connected(meta_args.layers, meta_args.nodes)]
        if meta_args.inverse:
            model += [tf.Variable(.5, trainable=True, name='inverse_var')]
    else:
        raise ValueError('Network type not understood:' + meta_args.network)

    # initialize optimizer
    if meta_args.optimizer == 'adam':
        optimizer = [tf.keras.optimizers.Adam(learning_rate=meta_args.lr)]
    elif meta_args.optimizer == 'sgd':
        optimizer = [tf.keras.optimizers.SGD(learning_rate=meta_args.lr)]
    else:
        raise ValueError('Optimizer type not understood:' + meta_args.optimizer)

    #if meta_args.inverse:
        #optimizer += [tf.keras.optimizers.Adam(learning_rate=meta_args.lr*10, beta_1=0.7, beta_2=0.9)]

    # initialize task
    if meta_args.task == 'helmholtz':
        task = Helmholtz(inverse_var=meta_args.inverse_var, inverse=meta_args.inverse)
    elif meta_args.task == 'burgers':
        task = Burgers(inverse_var=meta_args.inverse_var, inverse=meta_args.inverse)
    elif meta_args.task == 'kirchhoff':
        task = Kirchhoff(inverse_var=meta_args.inverse_var, inverse=meta_args.inverse)
    elif meta_args.task == 'mnist':
        task = MNIST(inverse_var=meta_args.inverse_var, inverse=meta_args.inverse)
    else:
        raise ValueError('Task type not understood:' + meta_args.task)

    num_b_losses = 1 if meta_args.aggregate_boundaries else task.num_b_losses

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
        args.update({"alpha": tf.constant(0., dtype=tf.float32)})

    elif meta_args.update_rule == 'relobalo':
        update_rule = relobalo
        args = {"lam"+str(i): tf.constant(1.) for i in range(num_b_losses+1)}
        args.update({"l"+str(i): tf.constant(1.) for i in range(num_b_losses+1)})
        args.update({"l0"+str(i): tf.constant(1.) for i in range(num_b_losses+1)})
        args.update({"T": tf.constant(meta_args.T, dtype=tf.float32)})
        rho = (np.random.uniform(size=meta_args.epochs+1) < meta_args.rho).astype(int).astype(np.float32)
        args.update({'rho': tf.constant(rho[0], dtype=tf.float32)})
        alpha = [tf.constant(meta_args.alpha, tf.float32)]*(meta_args.epochs+1)
        #alpha = [tf.constant(1., tf.float32), tf.constant(0., tf.float32)]+[tf.constant(meta_args.alpha, tf.float32)]
        args.update({"alpha": alpha[0]})

    elif meta_args.update_rule == 'gradnorm':
        update_rule = gradnorm
        args = {"l"+str(i): tf.constant(1.) for i in range(num_b_losses+1)}
        alpha = [tf.constant(0.)]+[tf.constant(meta_args.T, dtype=tf.float32)]*(meta_args.epochs+1)
        args.update({"alpha": tf.constant(alpha[0], dtype=tf.float32)})
        model += [GradNormArgs(nterms=num_b_losses+1, alpha=alpha)]
        optimizer += [tf.keras.optimizers.Adam(learning_rate=meta_args.lr)]

    else:
        raise ValueError('Update rule not understood:' + meta_args.update_rule)


    summary = []
    args_summary = []
    if isinstance(args, dict):
        args_keys = args.keys()
    best_loss = 1e9
    best_val_loss = 1e9
    cooldown = meta_args.patience
    meltdown = 0
    
    try:
        best_model = tf.keras.models.clone_model(model[0])
    except:
        best_model = model[0]
        print("cloning model failed, saving best model separately is not possible")

    x, y = task.training_batch(meta_args.batch_size)
    print('start training of', meta_args.task, 'in', meta_args.path)
    start = time()
    for epoch in range(meta_args.epochs):
        
        grads, f_loss, b_losses, oargs = update_rule(
            model, 
            task, 
            tf.constant(x, dtype=tf.float32), 
            tf.constant(y, dtype=tf.float32), 
            args,
            meta_args.aggregate_boundaries
        )

        parameters = model[0].trainable_variables
        if meta_args.inverse:
            parameters += [model[1]]
            #optimizer[1].apply_gradients(zip(grads[1], [model[1]]))
        optimizer[0].apply_gradients(zip(grads[0], parameters))
        if meta_args.update_rule == 'gradnorm':
            optimizer[-1].apply_gradients(zip(grads[-1], model[-1].trainable_variables))
            
        loss = f_loss + tf.reduce_sum(b_losses)            
        
        if not meta_args.update_rule == 'gradnorm' or epoch == 0:
            args = oargs.copy()
        if (meta_args.update_rule == 'gradnorm' or meta_args.update_rule == 'relobalo') and epoch == 0:
            for i in range(num_b_losses+1):
                args['l0'+str(i)] = ([f_loss]+b_losses)[i]
        if meta_args.resample:
            x, y = task.training_batch(meta_args.batch_size)
        if len(alpha) > 1:
            args['alpha'] = alpha[1]
            alpha = alpha[1:]
        if meta_args.update_rule == 'relobalo':
            args['rho'] = rho[1]
            rho = rho[1:]
                
        if epoch % 1000 == 0:
            x, y, u = task.validation_batch()

            if meta_args.inverse:
                val_loss0, val_loss = task.validation_loss(model, x, y, u)
                summary.append([loss, val_loss0, val_loss, f_loss]+b_losses+[model[1].numpy()])
            else:
                val_loss = task.validation_loss(model, x, y, u)
                summary.append([loss, val_loss, f_loss]+b_losses)

            loss, f_loss, val_loss = gpu_to_numpy(reduce_mean_all([loss, f_loss, val_loss]))
            b_losses  = gpu_to_numpy(b_losses)

            
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
                best_model.set_weights(model[0].get_weights())
            if cooldown <= 0 and epoch > meta_args.epochs/10:
                cooldown = meta_args.patience
                meltdown += 1
                if meta_args.verbose:
                    print(' reducing LR to', (optimizer[0].lr*meta_args.factor).numpy())
                for o in optimizer:
                    tf.keras.backend.set_value(o.lr, o.lr*meta_args.factor)
                model[0].set_weights(best_model.get_weights())
            if meltdown > 4 or time()-start > 3.75*3600:
                model[0] = best_model
                if meta_args.verbose:
                    print('early stopping')
                break
            cooldown -= 1
    
            # print evaluation metrics
            if meta_args.verbose:
                print(
                    "epoch {:<5}".format(epoch),
                    "loss {:<.3e}".format(best_loss), 
                    "val_loss {:<.3e}".format(val_loss),
                    "F {0:<.2e}".format(f_loss), 
                    "B", b_losses,
                    "ARGS" if not isinstance(args, dict) else list(zip(args_keys, e_args)),
                    strftime('%H:%M:%S', gmtime(time()-start))
                )

    # save results
    unique_path = create_directory(os.path.join('experiments', meta_args.path))
    meta_args.path = unique_path
    append_to_results((time()-start)/epoch*1000, meta_args, best_loss, best_val_loss)
    np.save(os.path.join(unique_path, 'summary'), summary)
    if args is not None:
        np.save(os.path.join(unique_path, 'args_summary'), args_summary)
    model[0].save(os.path.join(unique_path, 'model_'+str(i)))
    task.visualise(model, unique_path)
    return best_val_loss
        

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='experiments', type=str, help='path where to store the results')

parser.add_argument('--layers', default=1, type=int, help='number of layers')
parser.add_argument('--nodes', default=32, type=int, help='number of nodes')
parser.add_argument('--network', default='fc', type=str, help='type of network')

parser.add_argument('--optimizer', default='adam', type=str, help='type of optimizer')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--patience', default=3, type=int, help='how many evaluations without improvement to wait before reducing learning rate')
parser.add_argument('--factor', default=.1, type=float, help='multiplicative factor by which to reduce the learning rate')

parser.add_argument('--task', default='helmholtz', type=str, help='type of task to fit')
parser.add_argument('--inverse', action='store_true', help='solve inverse problem')
parser.add_argument('--inverse_var', default=None, type=float, help='target inverse variable')
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
