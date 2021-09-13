import tensorflow as tf
import numpy as np

@tf.function
def manual(model, optimizer, pde, x, y, args:dict, aggregate_boundaries=False):
    f_loss, b_losses = pde.calculate_loss(model, x, y, aggregate_boundaries, training=True)

    loss = args['lam'+str(0)]*f_loss + tf.reduce_sum([args['lam'+str(i+1)]*b_losses[i] for i in range(len(b_losses))])

    # update model
    grads = tf.gradients(loss, model.trainable_variables)
    #optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return grads, loss, b_losses, args


@tf.function
def lrannealing(model, optimizer, pde, x, y, args:dict, aggregate_boundaries=False):
    f_loss, b_losses = pde.calculate_loss(model, x, y, aggregate_boundaries, training=True)

    grad_f  = tf.gradients(f_loss,  model.trainable_variables, unconnected_gradients='zero')
    grad_bs = [tf.gradients(b_losses[i], model.trainable_variables, unconnected_gradients='zero') for i in range(len(b_losses))]

    # LR annealing
    mean_grad_f = tf.reduce_mean(tf.abs(tf.concat([tf.reshape(g, (-1,)) for g in grad_f], axis=-1)))
    lambs_hat = [mean_grad_f / (tf.reduce_mean(tf.abs(tf.concat([tf.reshape(g, (-1,)) for g in grad_bs[i]], axis=-1)))+1e-8) for i in range(len(b_losses))] # add small epsilon to prevent division by 0

    lambs = [args['alpha']*args['lam'+str(i)] + (1-args['alpha'])*lambs_hat[i] for i in range(len(b_losses))]
             
    scaled_grads = []
    for i in range(len(grad_f)):
        scaled_grads.append(grad_f[i] + \
            tf.reduce_sum(tf.stack([lambs[j]*grad_bs[j][i] for j in range(len(grad_bs))], axis=0), axis=0))

    # update model
    #optimizer.apply_gradients(zip(scaled_grads, model.trainable_variables))
    
    # update args
    args = args.copy()
    for i in range(len(b_losses)):
        args['lam'+str(i)] = lambs[i]

    return grads, f_loss, b_losses, args


@tf.function
def softadapt(model, optimizer, pde, x, y, args:dict, aggregate_boundaries=False):
    f_loss, b_losses = pde.calculate_loss(model, x, y, aggregate_boundaries, training=True)

    T = args['T']
    losses = [f_loss] + b_losses

    lambs_hat = tf.stop_gradient(tf.nn.softmax([losses[i]/(args['l'+str(i)]*T) for i in range(len(losses))])*tf.cast(len(losses), dtype=tf.float32))
    lambs = [args['alpha']*args['lam'+str(i)] + (1-args['alpha'])*lambs_hat[i] for i in range(len(losses))]

    loss = tf.reduce_sum([lambs[i]*losses[i] for i in range(len(losses))])

    # update model
    grads = tf.gradients(loss, model.trainable_variables)
    #optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # update args
    args = args.copy()
    for i in range(len(b_losses)+1):
        args['lam'+str(i)] = lambs[i]
        args['l'+str(i)] = losses[i]
    return grads, f_loss, b_losses, args


@tf.function
def softadapt_rnd_lookback(model, optimizer, pde, x, y, args:dict, aggregate_boundaries=False):
    f_loss, b_losses = pde.calculate_loss(model, x, y, aggregate_boundaries, training=True)

    T = args['T']
    losses = [f_loss] + b_losses

    lambs_hat = tf.stop_gradient(tf.nn.softmax([losses[i]/(args['l'+str(i)]*T) for i in range(len(losses))])*tf.cast(len(losses), dtype=tf.float32))
    lambs0_hat = tf.stop_gradient(tf.nn.softmax([losses[i]/(args['l0'+str(i)]*T) for i in range(len(losses))])*tf.cast(len(losses), dtype=tf.float32))
    lambs = [args['rho']*args['alpha']*args['lam'+str(i)] + (1-args['rho'])*args['alpha']*lambs0_hat[i] + (1-args['alpha'])*lambs_hat[i] for i in range(len(losses))]

    loss = tf.reduce_sum([lambs[i]*losses[i] for i in range(len(losses))])

    # update model
    grads = tf.gradients(loss, model.trainable_variables)
    #optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # update args
    args = args.copy()
    for i in range(len(b_losses)+1):
        args['lam'+str(i)] = lambs[i]
        args['l'+str(i)] = losses[i]
    return grads, f_loss, b_losses, args


@tf.function
def gradnorm(model, optimizers, pde, x, y, args, aggregate_boundaries=False):
    f_loss, b_losses = pde.calculate_loss(model[0], x, y, aggregate_boundaries, training=True)

    L_i = model[1]([f_loss]+b_losses)
    L_W = tf.reduce_sum(L_i)

    GiW = [tf.norm(tf.gradients(L_i[i], model[0].trainable_variables[-2])[0]) for i in range(len(b_losses)+1)]
    GiW_average = tf.reduce_mean(tf.stack(GiW, axis=0), axis=0)
    li_tilde = [li / args['l'+str(i)] for i, li in enumerate([f_loss]+b_losses)]
    li_tilde_average = tf.reduce_mean(tf.stack(li_tilde, axis=0), axis=0)
    Ri = [li / li_tilde_average for li in li_tilde]

    L_w = tf.reduce_sum(tf.math.abs(tf.stack([tf.norm(giw - tf.stop_gradient(GiW_average*ri**args['alpha'])) for giw, ri in zip(GiW, Ri)], axis=0)), axis=0)
    grad_L_w = tf.gradients(L_w, model[1].trainable_variables)
    #optimizers[1].apply_gradients(zip(grad_L_w, model[1].trainable_variables))
    grad_L_W = tf.gradients(L_W, model[0].trainable_variables)
    #optimizers[0].apply_gradients(zip(grad_L_W, model[0].trainable_variables))

    new_args = {'l'+str(i): l for i, l in enumerate([f_loss]+b_losses)}
    new_args.update({'alpha': args['alpha']})
    return (grad_L_w, grad_L_W), f_loss, b_losses, new_args