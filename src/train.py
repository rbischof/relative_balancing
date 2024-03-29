import os
import argparse
import numpy as np
import tensorflow as tf

from time import time, strftime, gmtime

from tasks.burgers import Burgers
from tasks.helmholtz import Helmholtz
from tasks.kirchhoff import Kirchhoff
from tasks.poisson_L import Poisson_L
from tasks.diffusion_sorption import DiffusionSorption1D
from tasks.allen_cahn import AllenCahn
from update_rules import manual, lrannealing, softadapt, relobralo, gradnorm
from models import fully_connected, GradNormArgs, autoencoder
from utils import gpu_to_numpy, reduce_mean_all, append_to_results, create_directory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('Tensorflow version', tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def train(meta_args):

    # initialize task
    if meta_args.task == 'helmholtz':
        task = Helmholtz(inverse_var=meta_args.inverse_var, inverse=meta_args.inverse)
    elif meta_args.task == 'burgers':
        task = Burgers(inverse_var=meta_args.inverse_var, inverse=meta_args.inverse)
    elif meta_args.task == 'kirchhoff':
        task = Kirchhoff(inverse_var=meta_args.inverse_var, inverse=meta_args.inverse)
    elif meta_args.task == 'allen_cahn':
        task = AllenCahn(inverse_var=meta_args.inverse_var, inverse=meta_args.inverse)
    elif meta_args.task == 'poisson_L':
        task = Poisson_L(inverse_var=meta_args.inverse_var, inverse=meta_args.inverse)
    elif meta_args.task == 'diffusion_sorption':
        task = DiffusionSorption1D(inverse_var=meta_args.inverse_var, inverse=meta_args.inverse)
    else:
        raise ValueError('Task type not understood:' + meta_args.task)

    # initialize network
    if meta_args.network == 'fc':   
        model = [fully_connected(nlayers=meta_args.layers, nnodes=meta_args.nodes, data_min=task.data_min, data_max=task.data_max)]
    else:
        raise ValueError('Network type not understood:' + meta_args.network)

    # initialize trainable variable to approximate PDE parameter
    if meta_args.inverse:
        model += [tf.Variable(.5, trainable=True, name='inverse_var')]

    # initialize optimizer
    if meta_args.optimizer == 'adam':
        optimizer = [tf.keras.optimizers.Adam(learning_rate=meta_args.lr)]
    elif meta_args.optimizer == 'sgd':
        optimizer = [tf.keras.optimizers.SGD(learning_rate=meta_args.lr)]
    else:
        raise ValueError('Optimizer type not understood:' + meta_args.optimizer)

    # initialize update rule
    if meta_args.update_rule == 'manual':
        update_rule = manual
        args = {"lam"+str(i): tf.constant(1.) for i in range(task.num_b_losses+1)}
        alpha = [1.]
        args.update({"alpha": tf.constant(alpha[0], dtype=tf.float32)})

    elif meta_args.update_rule == 'lrannealing':
        update_rule = lrannealing
        args = {"lam"+str(i): tf.constant(1.) for i in range(task.num_b_losses)}
        alpha = [tf.constant(meta_args.alpha, dtype=tf.float32)]*((meta_args.epochs+1))
        args.update({"alpha": tf.constant(alpha[0], dtype=tf.float32)})

    elif meta_args.update_rule == 'softadapt':
        update_rule = softadapt
        args = {"lam"+str(i): tf.constant(1.) for i in range(task.num_b_losses+1)}
        args.update({"l"+str(i): tf.constant(1.) for i in range(task.num_b_losses+1)})
        args.update({"T": tf.constant(meta_args.T, dtype=tf.float32)})
        alpha = [tf.constant(meta_args.alpha, dtype=tf.float32)]
        args.update({"alpha": tf.constant(0., dtype=tf.float32)})

    elif meta_args.update_rule == 'relobralo':
        update_rule = relobralo
        args = {"lam"+str(i): tf.constant(1.) for i in range(task.num_b_losses+1)}
        args.update({"l"+str(i): tf.constant(1.) for i in range(task.num_b_losses+1)})
        args.update({"l0"+str(i): tf.constant(1.) for i in range(task.num_b_losses+1)})
        args["T"] = tf.constant(meta_args.T, dtype=tf.float32)
        rho = (np.random.uniform(size=meta_args.epochs+1) < meta_args.rho).astype(int).astype(np.float32)
        args['rho'] = tf.constant(rho[0], dtype=tf.float32)
        alpha = [tf.constant(1., tf.float32), tf.constant(0., tf.float32)]+[tf.constant(meta_args.alpha, tf.float32)]
        args["alpha"] = alpha[0]

    elif meta_args.update_rule == 'gradnorm':
        update_rule = gradnorm
        args = {"l"+str(i): tf.constant(1.) for i in range(task.num_b_losses+1)}
        alpha = [tf.constant(0.)]+[tf.constant(meta_args.T, dtype=tf.float32)]*(meta_args.epochs+1)
        args.update({"alpha": tf.constant(alpha[0], dtype=tf.float32)})
        model += [GradNormArgs(nterms=task.num_b_losses+1, alpha=alpha)]
        optimizer += [tf.keras.optimizers.Adam(learning_rate=meta_args.lr)]

    else:
        raise ValueError('Update rule not understood:' + meta_args.update_rule)


    # inititalize logging
    summary = []
    args_summary = []
    train_summary = []
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
        if meta_args.resample:
            x, y = task.training_batch(meta_args.batch_size)
        
        grads, f_loss, b_losses, args = update_rule(
            model, 
            task, 
            tf.constant(x, dtype=tf.float32), 
            tf.constant(y, dtype=tf.float32), 
            args
        )

        parameters = model[0].trainable_variables
        if meta_args.inverse:
            parameters += [model[1]]
        optimizer[0].apply_gradients(zip(grads[0], parameters))

        if meta_args.update_rule == 'gradnorm':
            optimizer[-1].apply_gradients(zip(grads[-1], model[-1].trainable_variables))
            
        if (meta_args.update_rule == 'gradnorm' or meta_args.update_rule == 'relobralo') and epoch == 1:
            for i in range(task.num_b_losses+1):
                args['l0'+str(i)] = ([f_loss]+b_losses)[i]
        if len(alpha) > 1:
            args['alpha'] = alpha[1]
            alpha = alpha[1:]
        if meta_args.update_rule == 'relobralo':
            args['rho'] = rho[1]
            rho = rho[1:]

        train_summary.append((f_loss + tf.reduce_sum(b_losses)).numpy())

        # evaluate and log       
        if epoch % 1000 == 0:
            loss = np.mean(train_summary)
            
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
                e_args = {k: v for k, v in zip(args.keys(), gpu_to_numpy(args.values()))}
            else:
                e_args = {'args': gpu_to_numpy(args.variables())}
            args_summary.append(list(e_args.values()))
        
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
                    "loss {:<.3e}".format(loss), 
                    "val_loss {:<.3e}".format(val_loss),
                    "F {0:<.2e}".format(f_loss), 
                    "B", b_losses,
                    "ARGS" if not isinstance(args, dict) else list(e_args.items()),
                    strftime('%H:%M:%S', gmtime(time()-start))
                )
            
            train_summary = []

    # save results
    unique_path = create_directory(os.path.join('experiments', meta_args.path))
    print('saving model in', unique_path)
    meta_args.path = unique_path
    append_to_results((time()-start)/epoch*1000, meta_args, best_loss, best_val_loss)
    np.save(os.path.join(unique_path, 'summary'), summary)
    if args is not None:
        np.save(os.path.join(unique_path, 'args_summary'), args_summary)
    model[0].save(os.path.join(unique_path, 'model'))
    task.visualise(model, unique_path)
    return best_val_loss
        

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='experiments', type=str, help='path where to store the results')

parser.add_argument('--layers', default=1, type=int, help='number of layers')
parser.add_argument('--nodes', default=32, type=int, help='number of nodes')
parser.add_argument('--network', default='fc', type=str, help='type of network')

parser.add_argument('--optimizer', default='adam', type=str, help='type of optimizer')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--patience', default=4, type=int, help='how many evaluations without improvement to wait before reducing learning rate')
parser.add_argument('--factor', default=.1, type=float, help='multiplicative factor by which to reduce the learning rate')

parser.add_argument('--task', default='helmholtz', type=str, help='type of task to fit')
parser.add_argument('--inverse', action='store_true', help='solve inverse problem')
parser.add_argument('--inverse_var', default=None, type=float, help='target inverse variable')
parser.add_argument('--update_rule', default='manual', type=str, help='type of balancing')
parser.add_argument('--T', default=1., type=float, help='temperature parameter for softmax')
parser.add_argument('--alpha', default=.999, type=float, help='rate for exponential decay')
parser.add_argument('--rho', default=1., type=float, help='rate for exponential decay')

parser.add_argument('--epochs', default=100000, type=int, help='number of epochs')
parser.add_argument('--resample', action='store_true', help='resample datapoints or keep them fixed')
parser.add_argument('--batch_size', default=1024, type=int, help='number of sampled points in a batch')
parser.add_argument('--verbose', action='store_true', help='print progress to terminal')

args = parser.parse_args()
train(args)
