import tensorflow as tf
import tensor_summary as ts

flags = tf.app.flags
FLAGS = flags.FLAGS


def dummy_activation_function(x):
    return x


def init_weights(shape, initializer=None, name=None):
    if initializer is not None:
        return tf.get_variable(name, shape, initializer=initializer())
    else:
        return tf.Variable(tf.random_normal(shape, stddev=0.01))


def init_bias(shape, name):
    return tf.constant(0.1,
                       shape=shape,
                       name=name)


def conv2d_layer(x, filter_shape, keep_prob=None, activation_function=None, name=None):
    if keep_prob is None:
        keep_prob = 1

    if activation_function is None:
        activation_function = tf.nn.relu

    if name is None:
        name = "conv2d_layer"

    xavier_init = tf.contrib.layers.xavier_initializer

    name += "/"
    with tf.name_scope(name):
        w = init_weights(filter_shape, xavier_init, name + "w")
        conv = tf.nn.conv2d(x,
                            w,
                            strides=[1, 1, 1, 1],
                            padding="SAME",
                            name="conv",
                            )

        conv = activation_function(conv, name="relu")

        conv = tf.nn.dropout(conv, keep_prob=keep_prob, name="drop_out")

    return conv


def pooling2d_layer(input, keep_prob=None, name=None):
    if name is None:
        name = "pooling"

    if keep_prob is None:
        keep_prob = 1

    name += "/"
    with tf.name_scope(name):
        max_pooling = tf.nn.max_pool(input,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding="SAME",
                                     name="max_pooling")

        max_pooling = tf.nn.dropout(max_pooling,
                                    keep_prob=keep_prob,
                                    name="drop_out")
    return max_pooling


def layer_perceptron(X, input_shape, layer_width, name=None, drop_prob=None, activate_function=None):
    if name is None:
        name = "perceptron"

    if drop_prob is None:
        drop_prob = 1

    if activate_function is None:
        activate_function = tf.sigmoid

    name += "/"
    xavier_init = tf.contrib.layers.xavier_initializer
    with tf.name_scope(name):
        # W = tf.Variable(tf.random_normal(input_shape + layer_width))
        W = init_weights(input_shape + layer_width,
                         initializer=xavier_init,
                         name=name+"weight")

        bias = init_bias(layer_width,
                         name=name+"bias")

        activate = activate_function(tf.matmul(X, W) + bias)

        activate = tf.nn.dropout(activate,
                                 keep_prob=drop_prob,
                                 name="drop_out")

        ts.variable_summaries(W)
        ts.variable_summaries(bias)

    return activate


def placeholders_init(name=None):
    if name is None:
        name = "input"

    # placeHolder
    with tf.name_scope(name):
        x = tf.placeholder(tf.float32, [None, FLAGS.image_size], name="X")
        y = tf.placeholder(tf.float32, [None, FLAGS.label_number], name="Y")
        y_label = tf.placeholder(tf.float32, [None], name="Y_label")

    ph_set = {"X": x,
              "Y": y,
              "Y_label": y_label}

    return ph_set
