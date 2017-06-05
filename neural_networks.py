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


def init_bias(shape):
    return tf.constant(0.1, shape=shape)


def conv2d_layer(x, w, b, keep_prob=None, activation_function=None, name=None):
    if keep_prob is None:
        keep_prob = 1

    if activation_function is None:
        activation_function = tf.nn.relu

    if name is None:
        name = "convolution"

    with tf.name_scope(name):
        conv = tf.nn.conv2d(x,
                            w,
                            strides=[1, 1, 1, 1],
                            padding="SAME")

    with tf.name_scope("activate"):
        conv_a = activation_function(conv + b)

    with tf.name_scope("pooling"):
        max_pool = tf.nn.max_pool(conv_a,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding="SAME")

    dropout = tf.nn.dropout(max_pool, keep_prob=keep_prob)

    return dropout


def layer_perceptron(X, input_shape, layer_width, layer_name=None, drop_prob=None, activate_function=None):
    if layer_name is None:
        layer_name = "layer"

    if drop_prob is None:
        drop_prob = 1

    if activate_function is None:
        activate_function = tf.sigmoid

    xavier_init = tf.contrib.layers.xavier_initializer
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            # W = tf.Variable(tf.random_normal(input_shape + layer_width))
            W = init_weights(input_shape + layer_width,
                             initializer=xavier_init, name=layer_name+"weights")
            ts.variable_summaries(W)

        with tf.name_scope("bias"):
            bias = init_bias(layer_width)
            ts.variable_summaries(bias)

        with tf.name_scope("activate_function"):
            activate = activate_function(tf.matmul(X, W) + bias)

        activate = tf.nn.dropout(activate, keep_prob=drop_prob)

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
