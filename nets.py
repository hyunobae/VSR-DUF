## -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]


def FR_16L(x, is_train, uf=4):
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1, 3, 3, 3, 64], [1, 1, 1, 1, 1], 'VALID', name='conv1')

    F = 64
    G = 32
    for r in range(3):
        t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

        t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, stp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
                   name='Rconv' + str(r + 1) + 'b')

        x = tf.concat([x, t], 4)
        F += G
    for r in range(3, 6):
        t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

        t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, sp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
                   name='Rconv' + str(r + 1) + 'b')

        x = tf.concat([x[:, 1:-1], t], 4)
        F += G

    x = BatchNorm(x, is_train, name='fbn1')
    x = tf.nn.relu(x)
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1, 3, 3, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='conv2')
    x = tf.nn.relu(x)

    r = Conv3D(x, [1, 1, 1, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='rconv1')
    r = tf.nn.relu(r)
    r = Conv3D(r, [1, 1, 1, 256, 3 * uf * uf], [1, 1, 1, 1, 1], 'VALID', name='rconv2')

    f = Conv3D(x, [1, 1, 1, 256, 512], [1, 1, 1, 1, 1], 'VALID', name='fconv1')
    f = tf.nn.relu(f)
    f = Conv3D(f, [1, 1, 1, 512, 1 * 5 * 5 * uf * uf], [1, 1, 1, 1, 1], 'VALID', name='fconv2')

    ds_f = tf.shape(f)
    f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, uf * uf])
    f = tf.nn.softmax(f, dim=4)

    return f, r


def FR_28L(x, is_train, uf=4):
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1, 3, 3, 3, 64], [1, 1, 1, 1, 1], 'VALID', name='conv1')

    F = 64
    G = 16
    for r in range(9):
        t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

        t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, stp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
                   name='Rconv' + str(r + 1) + 'b')

        x = tf.concat([x, t], 4)
        F += G
    for r in range(9, 12):
        t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

        t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, sp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
                   name='Rconv' + str(r + 1) + 'b')

        x = tf.concat([x[:, 1:-1], t], 4)
        F += G

    x = BatchNorm(x, is_train, name='fbn1')
    x = tf.nn.relu(x)
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1, 3, 3, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='conv2')

    x = tf.nn.relu(x)

    r = Conv3D(x, [1, 1, 1, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='rconv1')
    r = tf.nn.relu(r)
    r = Conv3D(r, [1, 1, 1, 256, 3 * uf * uf], [1, 1, 1, 1, 1], 'VALID', name='rconv2')

    f = Conv3D(x, [1, 1, 1, 256, 512], [1, 1, 1, 1, 1], 'VALID', name='fconv1')
    f = tf.nn.relu(f)
    f = Conv3D(f, [1, 1, 1, 512, 1 * 5 * 5 * uf * uf], [1, 1, 1, 1, 1], 'VALID', name='fconv2')

    ds_f = tf.shape(f)
    f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, uf * uf])
    f = tf.nn.softmax(f, dim=4)

    return f, r


def FR_52L(x, is_train, uf=4):
    inputs = Input(shape=x.shape)
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1, 3, 3, 3, 64], [1, 1, 1, 1, 1], 'VALID', name='conv1')

    F = 64
    G = 16
    for r in range(0, 21):
        t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

        t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, stp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
                   name='Rconv' + str(r + 1) + 'b')

        x = tf.concat([x, t], 4)
        F += G
    for r in range(21, 24):
        t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

        t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, sp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
                   name='Rconv' + str(r + 1) + 'b')

        x = tf.concat([x[:, 1:-1], t], 4)
        F += G

    x = BatchNorm(x, is_train, name='fbn1')
    x = tf.nn.relu(x)
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1, 3, 3, 448, 256], [1, 1, 1, 1, 1], 'VALID', name='conv2')

    x = tf.nn.relu(x)

    r = Conv3D(x, [1, 1, 1, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='rconv1')
    r = tf.nn.relu(r)
    r = Conv3D(r, [1, 1, 1, 256, 3 * uf * uf], [1, 1, 1, 1, 1], 'VALID', name='rconv2')

    f = Conv3D(x, [1, 1, 1, 256, 512], [1, 1, 1, 1, 1], 'VALID', name='fconv1')
    f = tf.nn.relu(f)
    f = Conv3D(f, [1, 1, 1, 512, 1 * 5 * 5 * uf * uf], [1, 1, 1, 1, 1], 'VALID', name='fconv2')

    ds_f = tf.shape(f)
    f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, uf * uf])
    f = tf.nn.softmax(f, dim=4)

    model_residual = Model(inputs, r)
    model_filter = Model(inputs, f)
    return model_filter, model_residual

he_normal_init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')
def BatchNorm(input, is_train, decay=0.999, name='BatchNorm'):
    '''
    https://github.com/zsdonghao/tensorlayer/blob/master/tensorlayer/layers.py
    https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
    http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow
    '''
    from tensorflow.python.training import moving_averages
    from tensorflow.python.ops import control_flow_ops

    axis = list(range(len(input.get_shape()) - 1))
    fdim = input.get_shape()[-1:]

    with tf.variable_scope(name):
        beta = tf.get_variable('beta', fdim, initializer=tf.constant_initializer(value=0.0))
        gamma = tf.get_variable('gamma', fdim, initializer=tf.constant_initializer(value=1.0))
        moving_mean = tf.get_variable('moving_mean', fdim, initializer=tf.constant_initializer(value=0.0),
                                      trainable=False)
        moving_variance = tf.get_variable('moving_variance', fdim, initializer=tf.constant_initializer(value=0.0),
                                          trainable=False)

        def mean_var_with_update():
            batch_mean, batch_variance = tf.nn.moments(input, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=True)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_variance, decay,
                                                                           zero_debias=True)
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        mean, variance = control_flow_ops.cond(is_train, mean_var_with_update, lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(input, mean, variance, beta, gamma,
                                     1e-3)  # , tf.stack([mean[0], variance[0], beta[0], gamma[0]])


def Conv3D(input, kernel_shape, strides, padding, name='Conv3d', W_initializer=he_normal_init, bias=True):
    with tf.variable_scope(name):
        W = tf.get_variable("W", kernel_shape, initializer=W_initializer)
        if bias is True:
            b = tf.get_variable("b", (kernel_shape[-1]), initializer=tf.constant_initializer(value=0.0))
        else:
            b = 0

    return tf.nn.conv3d(input, W, strides, padding) + b
