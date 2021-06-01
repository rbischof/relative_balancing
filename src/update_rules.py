import tensorflow as tf

@tf.function
def manual(model, optimizer, problem, x, y, u, args:list=None):
    f_loss, b_losses, val_loss = problem.calculate_loss(x, y, u, training=True)

    if args == None:
        args = [1.]*(len(b_losses)+1)
    assert (len(b_losses)+1) == len(args)

    loss = args[0]*f_loss + tf.reduce_sum([args[i+1]*b_losses[i] for i in range(len(b_losses))])

    # update model
    grads = tf.gradients(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, b_losses, val_loss, args


@tf.function
def lrannealing(model, optimizer, problem, x, y, u, args:list):
    f_loss, b_losses, val_loss = problem.calculate_loss(x, y, u, training=True)

    alpha = args['alpha'][0]

    grad_f  = tf.gradients(f_loss,  model.trainable_variables, unconnected_gradients='zero')
    grad_bs = [tf.gradients(b_losses[i], model.trainable_variables, unconnected_gradients='zero') for i in range(len(b_losses))]

    # LR annealing
    mean_grad_f = tf.reduce_mean(tf.abs(tf.concat([tf.reshape(g, (-1,)) for g in grad_f], axis=-1)))
    lambs_hat = [mean_grad_f / (tf.reduce_mean(tf.abs(tf.concat([tf.reshape(g, (-1,)) for g in grad_bs[i]], axis=-1)))+1e-8) for i in range(len(b_losses))] # add small epsilon to prevent division by 0

    lambs = [alpha*args['lam'+str(i)] + (1-alpha)*lambs_hat[i] for i in range(len(b_losses))]
             
    scaled_grads = []
    for i in range(len(grad_f)):
        scaled_grads.append(grad_f[i] + \
            tf.reduce_sum(tf.stack([lambs[i]*grad_bs[i] for i in range(len(b_losses))], axis=0), axis=0))

    # update model
    optimizer.apply_gradients(zip(scaled_grads, model.trainable_variables))

    # update args
    args['alpha'] = args['alpha'][1:]
    return f_loss, b_losses, val_loss, args


@tf.function
def relative(model, optimizer, problem, x, y, u, args):
    f_loss, b_losses, val_loss = problem.calculate_loss(x, y, u, training=True)

    alpha = args['alpha'][0]
    T = args['T']
    losses = [f_loss] + b_losses

    lambs_hat = tf.stop_gradient(tf.nn.softmax([losses[i]/(args['l'+str(i)]*T) for i in range(len(losses))])*tf.cast(len(losses), dtype=tf.float32))
    lambs = [alpha*args['lam'+str(i)] + (1-alpha)*lambs_hat[i] for i in range(len(b_losses)+1)]

    loss = tf.reduce_sum([lambs[i]*losses[i] for i in range(len(b_losses))])

    # update model
    grads = tf.gradients(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # update args
    args['alpha'] = args['alpha'][1:]
    for i in range(len(b_losses)+1):
        args['lam'+str(i)] = lambs[i]
        args['l'+str(i)] = losses[i]
    return f_loss, b_losses, val_loss, args
    