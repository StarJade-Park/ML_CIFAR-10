import tensorflow as tf
import tensor_summary as ts
from tensorflow.contrib.layers import xavier_initializer

flags = tf.app.flags
FLAGS = flags.FLAGS


def dummy_activation_function(x):
    return x


def init_weights(shape, initializer=None, name=None, decay=None):
    if initializer is not None:
        var = tf.get_variable(name,
                              shape,
                              initializer=xavier_initializer())
    else:
        var = tf.Variable(tf.random_normal(shape, mean=0.2, stddev=0.01))

    if decay is not None:
        var = tf.multiply(var, 1)
    return var


def init_bias(shape, name):
    return tf.constant(0.0,
                       shape=shape,
                       name=name)


# https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
def batch_norm_wrapper(inputs, is_training, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    epsilon = 1e-7

    if is_training is True:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, epsilon)


def conv2d_layer(x, w, is_training, keep_prob=1, activation_function=None, name="conv2d_layer"):
    if activation_function is None:
        activation_function = tf.nn.relu

    with tf.name_scope("conv2d"):
        conv = tf.nn.conv2d(x,
                            w,
                            strides=[1, 1, 1, 1],
                            padding="SAME",
                            name="conv",
                            )

    with tf.name_scope("normalization"):
        conv = batch_norm_wrapper(conv, is_training=is_training)

    with tf.name_scope("activate"):
        conv = activation_function(conv, name="activate")

    with tf.name_scope("dropout"):
        conv = tf.nn.dropout(conv, keep_prob=keep_prob, name="drop_out")

    # print(conv)
    return conv


def pooling2d_layer(input, keep_prob=1, name="pooling"):
    with tf.name_scope("max_pooling"):
        max_pooling = tf.nn.max_pool(input,
                                     ksize=[1, 3, 3, 1],
                                     strides=[1, 2, 2, 1],
                                     padding="SAME",
                                     name="max_pooling")

    with tf.name_scope("dropout"):
        dropout = tf.nn.dropout(max_pooling,
                                keep_prob=keep_prob,
                                name="drop_out")

    return dropout


def layer_perceptron(X, W, bias, is_training, name="perceptron",
                     drop_prob=1, activate_function=None):
    with tf.name_scope("pre_activate"):
        pre_activate = tf.add(tf.matmul(X, W), bias)

    with tf.name_scope("normalization"):
        norm = batch_norm_wrapper(pre_activate, is_training)

    with tf.name_scope("activate"):
        if activate_function is "relu":
            activate = tf.nn.relu(norm)
        else:
            activate = tf.nn.sigmoid(norm)

        activate = tf.nn.dropout(activate,
                                 keep_prob=drop_prob,
                                 name="drop_out")

    ts.variable_summaries(W)
    ts.variable_summaries(bias)

    return activate


def placeholders_init(size, name="input"):
    # placeHolder
    with tf.name_scope("input_X"):
        x = tf.placeholder(tf.float32, [None, size * size * 3], name="X")

    with tf.name_scope("input_Y"):
        y = tf.placeholder(tf.float32, [None, 10], name="Y")

    with tf.name_scope("input_label"):
        y_label = tf.placeholder(tf.float32, [None], name="Y_label")

    ph_set = {"X": x,
              "Y": y,
              "Y_label": y_label}

    return ph_set
