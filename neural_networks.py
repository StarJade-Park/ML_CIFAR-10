import tensorflow as tf
import tensor_summary as ts

flags = tf.app.flags
FLAGS = flags.FLAGS


def layer_perceptron(X, input_shape, layer_width, layer_name=None):
    if layer_name is None:
        layer_name = "layer"


    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            W = tf.Variable(tf.random_normal(input_shape + layer_width))
            ts.variable_summaries(W)
        with tf.name_scope("bias"):
            bias = tf.Variable(tf.random_normal(layer_width))
            ts.variable_summaries(bias)
        with tf.name_scope("activate_function"):
            activate = tf.sigmoid(tf.matmul(X, W) + bias)

    return activate


def placeholders_init(name=None):
    if name is None:
        name = "input"
    # placeHolder
    with tf.name_scope(name):
        x = tf.placeholder(tf.float32, [None, FLAGS.image_size], name="X")
        y = tf.placeholder(tf.float32, [None, FLAGS.label_number], name="Y")
        y_label = tf.placeholder(tf.float32, [None], name="Y_label")

    ph_set = {"X": x, "Y": y, "Y_label": y_label}

    return ph_set
