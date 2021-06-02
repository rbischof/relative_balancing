import tensorflow as tf

@tf.function
def manual(model, optimizer, pde, x, y, u, args:dict=None, alpha=None):
    f_loss, b_losses, val_loss = pde.calculate_loss(model, x, y, u, training=True)

    if args == None:
        args = [1.]*(len(b_losses)+1)

    loss = args['lam'+str(0)]*f_loss + tf.reduce_sum([args['lam'+str(i+1)]*b_losses[i] for i in range(len(b_losses))])

    # update model
    grads = tf.gradients(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, b_losses, val_loss, args


@tf.function
def lrannealing(model, optimizer, pde, x, y, u, args:dict, alpha=None):
    f_loss, b_losses, val_loss = pde.calculate_loss(model, x, y, u, training=True)

    grad_f  = tf.gradients(f_loss,  model.trainable_variables, unconnected_gradients='zero')
    grad_bs = [tf.gradients(b_losses[i], model.trainable_variables, unconnected_gradients='zero') for i in range(len(b_losses))]

    # LR annealing
    mean_grad_f = tf.reduce_mean(tf.abs(tf.concat([tf.reshape(g, (-1,)) for g in grad_f], axis=-1)))
    lambs_hat = [mean_grad_f / (tf.reduce_mean(tf.abs(tf.concat([tf.reshape(g, (-1,)) for g in grad_bs[i]], axis=-1)))+1e-8) for i in range(len(b_losses))] # add small epsilon to prevent division by 0

    lambs = [alpha*args['lam'+str(i)] + (1-alpha)*lambs_hat[i] for i in range(len(b_losses))]
             
    scaled_grads = []
    for i in range(len(grad_f)):
        scaled_grads.append(grad_f[i] + \
            tf.reduce_sum(tf.stack([lambs[j]*grad_bs[j][i] for j in range(len(grad_bs[i]))], axis=0), axis=0))

    # update model
    optimizer.apply_gradients(zip(scaled_grads, model.trainable_variables))

    return f_loss, b_losses, val_loss, args


@tf.function
def relative(model, optimizer, pde, x, y, u, args:dict, alpha=None):
    f_loss, b_losses, val_loss = pde.calculate_loss(model, x, y, u, training=True)

    T = args['T']
    losses = [f_loss] + b_losses

    lambs_hat = tf.stop_gradient(tf.nn.softmax([losses[i]/(args['l'+str(i)]*T) for i in range(len(losses))])*tf.cast(len(losses), dtype=tf.float32))
    lambs = [alpha*args['lam'+str(i)] + (1-alpha)*lambs_hat[i] for i in range(len(b_losses)+1)]

    loss = tf.reduce_sum([lambs[i]*losses[i] for i in range(len(b_losses))])

    # update model
    grads = tf.gradients(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # update args
    args = args.copy()
    for i in range(len(b_losses)+1):
        args['lam'+str(i)] = lambs[i]
        args['l'+str(i)] = losses[i]
    return f_loss, b_losses, val_loss, args


@tf.function
def grad_norm(model, optimizer, pde, x, y, u, args:dict):
    f_loss, b_losses, val_loss = pde.calculate_loss(model, x, y, u, training=True)

    loss = tf.reduce_sum([f_loss*args['w0']] + [b_losses[i]*args['w'+str(i+1)] for i in range(len(b_losses)+1)])

    GiW = [tf.norm(tf.gradients(args['w'+str(i)]*Li, model.trainable_variables[-2])[0]) for i, Li in enumerate([f_loss]+b_losses)]
    GiW_average = tf.reduce_mean(tf.stack(GiW, axis=0), axis=0)
    li_tilde = [Li / args['l'+str(i)] for i, Li in enumerate([f_loss]+b_losses)]
    li_tilde_average = tf.reduce_mean(tf.stack(li_tilde, axis=0), axis=0)
    Ri = [Li / li_tilde_average for Li in li_tilde]

    L_grad = tf.reduce_sum(tf.stack([tf.norm(giw - tf.stop_gradient(GiW_average*ri**args['alpha'])) for giw, ri in zip(GiW, Ri)], axis=0), axis=0)
    T = args['T']
    losses = [f_loss] + b_losses

    return f_loss, b_losses, val_loss, args