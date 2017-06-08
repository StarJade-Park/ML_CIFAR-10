import tensorflow as tf
import tensor_summary as ts
import neural_networks as nn
import random
import pickle

import os
import all_flags

# TODO file must split
# TODO write more comment please

flags = tf.app.flags
FLAGS = flags.FLAGS


# TODO add argv for modeling function ex) layer width, layer number
def model_NN():
    # placeholder x, y, y_label
    ph_set = nn.placeholders_init()

    # NN layer

    layer1 = nn.layer_perceptron(ph_set["X"],
                                 [FLAGS.image_size],
                                 [FLAGS.perceptron_input_shape_size],
                                 "layer_1")
    layer2 = nn.layer_perceptron(layer1,
                                 [FLAGS.perceptron_input_shape_size],
                                 [FLAGS.perceptron_output_shape_size],
                                 "layer_2")
    h = nn.layer_perceptron(layer2, [FLAGS.perceptron_input_shape_size],
                            [FLAGS.label_number], "layer_3")

    # cost function
    with tf.name_scope("cost_function"):
        # TODO LOOK ME logistic regression does not work, for now use square error method
        # cost function square error method
        cost = tf.reduce_mean((h - ph_set["Y"]) ** 2, name="cost")
        # logistic regression
        # cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h), name="cost")
        ts.variable_summaries(cost)

    # train op
    with tf.name_scope("train_op"):
        train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)
        # if train_op is None:
        #     pass
        # tf.summary.histogram(train_op)

    # predicted label and batch batch_acc
    predicted_label = tf.cast(tf.arg_max(h, 1, name="predicted_label"), tf.float32)
    with tf.name_scope("NN_batch_acc"):
        batch_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_label, ph_set["Y_label"]), tf.float32),
                                   name="batch_acc")
        tf.summary.scalar("accuracy", batch_acc)
        batch_hit_count = tf.reduce_sum(tf.cast(tf.equal(predicted_label, ph_set["Y_label"]), tf.float32),
                                        name="batch_hit_count")
        tf.summary.scalar("hit_count", batch_hit_count)

    # merge summary
    summary = tf.summary.merge_all()

    # init op
    init_op = tf.global_variables_initializer()

    # save tensor
    tensor_set = {"X": ph_set["X"],
                  "Y": ph_set["Y"],
                  "Y_label": ph_set["Y_label"],
                  "layer1": layer1,
                  "layer2": layer2,
                  "h": h,
                  "cost": cost,
                  "train_op": train_op,
                  "predicted_label": predicted_label,
                  "batch_acc": batch_acc,
                  "batch_hit_count ": batch_hit_count,
                  "init_op": init_op,
                  "summary": summary,
                  }

    return tensor_set


# TODO add argv for modeling function ex) layer width, layer number
def model_NN_softmax():
    # placeHolder
    ph_set = nn.placeholders_init()

    # NN layer
    layer1 = nn.layer_perceptron(ph_set["X"], [FLAGS.image_size],
                                 [FLAGS.perceptron_output_shape_size], "softmax_L1")
    layer2 = nn.layer_perceptron(layer1, [FLAGS.perceptron_input_shape_size],
                                 [FLAGS.perceptron_output_shape_size], "softmax_L2")
    layer3 = nn.layer_perceptron(layer2, [FLAGS.perceptron_input_shape_size],
                                 [FLAGS.label_number], "softmax_L3")

    # softmax layer
    with tf.name_scope("softmax_func"):
        W_softmax = tf.Variable(tf.zeros([10, 10]), name="W_softmax")
        h = tf.nn.softmax(tf.matmul(layer3, W_softmax), name="h")
        ts.variable_summaries(h)

    # cross entropy function
    with tf.name_scope("cross_entropy"):
        cost = tf.reduce_mean(-tf.reduce_sum(ph_set["Y"] * tf.log(h), reduction_indices=1))
        ts.variable_summaries(cost)

    # train op
    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)

    # predicted label and batch batch_acc
    predicted_label = tf.cast(tf.arg_max(h, 1, name="predicted_label"), tf.float32)
    with tf.name_scope("softmax_batch_acc"):
        with tf.name_scope("accuracy"):
            batch_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_label, ph_set["Y_label"]), tf.float32),
                                       name="batch_acc")
            tf.summary.scalar("accuracy", batch_acc)

        with tf.name_scope("batch_hit_count"):
            batch_hit_count = tf.reduce_sum(tf.cast(tf.equal(predicted_label, ph_set["Y_label"]), tf.float32),
                                            name="batch_hit_count")
            tf.summary.scalar("hit_count", batch_hit_count)

    # merge summary
    summary = tf.summary.merge_all()

    # init op
    init_op = tf.global_variables_initializer()

    # save tensor
    tensor_set = {"X": ph_set["X"],
                  "Y": ph_set["Y"],
                  "Y_label": ph_set["Y_label"],
                  "layer1": layer1,
                  "layer2": layer2,
                  "layer3": layer3,
                  "W_softmax ": W_softmax,
                  "h": h,
                  "cost": cost,
                  "train_op": train_op,
                  "predicted_label": predicted_label,
                  "batch_acc": batch_acc,
                  "batch_hit_count ": batch_hit_count,
                  "init_op": init_op,
                  "summary": summary,
                  }
    return tensor_set


class Model_cnn_nn_softmax:
    PARAM_LIST = ["epoch_size",
                  "train_size",
                  "mini_batch_size",
                  "conv_dropout_rate",
                  "fc_dropout_rate",
                  "initializer",
                  "conv1_out",
                  "conv1_filter_size",
                  "conv1_w_shape",
                  "conv2_out",
                  "conv2_filter_size",
                  "conv2_w_shape",
                  "conv3_out",
                  "conv3_filter_size",
                  "conv3_w_shape",
                  "fc_layer_depth",
                  "fc_layer_width",
                  "fc_input_size",
                  "fc_layer1_w_shape",
                  "fc_layer1_bias_shape",
                  "fc_layer2_w_shape",
                  "fc_layer2_bias_shape",
                  "softmax_w_shape",
                  "l2_regularizer_beta",
                  "learning_rate",
                  ]

    def __init__(self):
        pass

    def default_param(self):
        param = {}
        param["param_list"] = self.PARAM_LIST

        param["epoch_size"] = 20000

        param["train_size"] = 500
        param["mini_batch_size"] = 50
        param["conv_dropout_rate"] = 0.7
        param["fc_dropout_rate"] = 0.5
        param["initializer"] = tf.contrib.layers.xavier_initializer

        param["conv1_out"] = 32
        param["conv1_filter_size"] = 3
        param["conv1_w_shape"] = [param["conv1_filter_size"],
                                  param["conv1_filter_size"],
                                  3,
                                  param["conv1_out"]]

        param["conv2_out"] = 32
        param["conv2_filter_size"] = 3
        param["conv2_w_shape"] = [param["conv2_filter_size"],
                                  param["conv2_filter_size"],
                                  param["conv1_out"],
                                  param["conv2_out"]]

        param["conv3_out"] = 32
        param["conv3_filter_size"] = 3
        param["conv3_w_shape"] = [param["conv3_filter_size"],
                                  param["conv3_filter_size"],
                                  param["conv2_out"],
                                  param["conv3_out"]]

        param["fc_layer_depth"] = 2
        param["fc_layer_width"] = 128
        param["fc_input_size"] = 4 * 4 * param["conv3_out"]
        param["fc_layer1_w_shape"] = [4 * 4 * param["conv3_out"], param["fc_layer_width"]]
        param["fc_layer1_bias_shape"] = [param["fc_layer_width"]]
        param["fc_layer2_w_shape"] = [param["fc_layer_width"], param["fc_layer_width"]]
        param["fc_layer2_bias_shape"] = [param["fc_layer_width"]]
        param["softmax_w_shape"] = [param["fc_layer_width"], 10]
        param["l2_regularizer_beta"] = 0.5
        param["learning_rate"] = 1e-4

        param["param_list"] = self.PARAM_LIST

        return param

    def build_model(self, param):
        ph_set = nn.placeholders_init()

        x_ = tf.reshape(ph_set["X"], [-1, 32, 32, 3])

        is_training = tf.placeholder(tf.bool, name="is_training")

        global_step = tf.Variable(initial_value=0,
                                  name='global_step', trainable=False)

        global_epoch = tf.Variable(initial_value=0,
                                   name='global_epoch', trainable=False)

        inc_global_epoch = tf.assign(global_epoch, global_epoch + 1)


        conv_dropout_rate = tf.placeholder(tf.float32, name="conv_dropout_rate")
        fc_dropout_rate = tf.placeholder(tf.float32, name="fc_dropout_rate")

        # conv layer 1,2,3
        # conv1 = nn.conv2d_layer(x_,
        #                         [3, 3, 3, 64],
        #                         keep_prob=conv_dropout_rate,
        #                         name="conv1")
        #
        # conv2 = nn.conv2d_layer(conv1,
        #                         [3, 3, 64, 64],
        #                         keep_prob=conv_dropout_rate,
        #                         name="conv2")
        #
        # pooling1 = nn.pooling2d_layer(conv2,
        #                               keep_prob=1,
        #                               name="pooling1")
        #
        # conv3 = nn.conv2d_layer(pooling1,
        #                         [3, 3, 64, 128],
        #                         keep_prob=conv_dropout_rate,
        #                         name="conv3")
        #
        # conv4 = nn.conv2d_layer(conv3,
        #                         [3, 3, 128, 128],
        #                         keep_prob=1,
        #                         name="conv4")
        #
        # pooling2 = nn.pooling2d_layer(conv4,
        #                               keep_prob=conv_dropout_rate,
        #                               name="pooling2")
        #
        # conv5 = nn.conv2d_layer(pooling2,
        #                         [3, 3, 128, 256],
        #                         keep_prob=conv_dropout_rate,
        #                         name="conv5")
        #
        # conv6 = nn.conv2d_layer(conv5,
        #                         [3, 3, 256, 256],
        #                         keep_prob=conv_dropout_rate,
        #                         name="conv6")
        #
        # conv7 = nn.conv2d_layer(conv6,
        #                         [3, 3, 256, 256],
        #                         keep_prob=1,
        #                         name="conv7")
        #
        # pooling3 = nn.pooling2d_layer(conv7,
        #                               keep_prob=conv_dropout_rate,
        #                               name="pooling3")
        #
        # conv8 = nn.conv2d_layer(pooling3,
        #                         [3, 3, 256, 512],
        #                         keep_prob=conv_dropout_rate,
        #                         name="conv8")
        #
        # conv9 = nn.conv2d_layer(conv8,
        #                         [3, 3, 512, 512],
        #                         keep_prob=conv_dropout_rate,
        #                         name="conv9")
        #
        # conv10 = nn.conv2d_layer(conv9,
        #                          [3, 3, 512, 512],
        #                          keep_prob=1,
        #                          name="conv10")
        #
        # pooling4 = nn.pooling2d_layer(conv10,
        #                               keep_prob=conv_dropout_rate,
        #                               name="pooling4")
        #
        # conv11 = nn.conv2d_layer(pooling4,
        #                          [3, 3, 512, 512],
        #                          keep_prob=conv_dropout_rate,
        #                          name="conv11")
        #
        # conv12 = nn.conv2d_layer(conv11,
        #                          [3, 3, 512, 512],
        #                          keep_prob=conv_dropout_rate,
        #                          name="conv12")
        #
        # conv13 = nn.conv2d_layer(conv12,
        #                          [3, 3, 512, 512],
        #                          keep_prob=1,
        #                          name="conv13")
        #
        # pooling5 = nn.pooling2d_layer(conv13,
        #                               keep_prob=conv_dropout_rate,
        #                               name="pooling5")

        # full connect layer

        initializer = param["initializer"]

        conv1_w = nn.init_weights(param["conv1_w_shape"], initializer, "conv1/w")
        conv1 = nn.conv2d_layer(x_,
                                conv1_w,
                                is_training,
                                keep_prob=conv_dropout_rate,
                                name="conv1")

        pooling1 = nn.pooling2d_layer(conv1, name="pooling1")

        conv2_w = nn.init_weights(param["conv2_w_shape"], initializer, "conv2/w")
        conv2 = nn.conv2d_layer(pooling1,
                                conv2_w,
                                is_training,
                                keep_prob=conv_dropout_rate,
                                name="conv2")

        pooling2 = nn.pooling2d_layer(conv2, name="pooling2")

        conv3_w = nn.init_weights(param["conv3_w_shape"], initializer, "conv3/w")
        conv3 = nn.conv2d_layer(pooling2,
                                conv3_w,
                                is_training,
                                keep_prob=conv_dropout_rate,
                                name="conv3")

        pooling3 = nn.pooling2d_layer(conv3, name="pooling2")

        # print(tf.shape(pooling3))
        fc_input_size = param["fc_input_size"]
        # fc_input_size = 32*32*3
        conv_out_reshape = tf.reshape(pooling3, [-1, fc_input_size])

        layer_width = param["fc_layer_width"]
        # print(conv_out_reshape)
        activate_function = tf.nn.relu
        with tf.name_scope("fc_layer"):
            fc_layer1_w = nn.init_weights(param["fc_layer1_w_shape"],
                                          initializer=initializer,
                                          name="fc_layer1/weight")
            fc_layer1_bias = nn.init_bias(param["fc_layer1_bias_shape"], name="fc_layer1/bias")
            fc_layer1 = nn.layer_perceptron(conv_out_reshape,
                                            fc_layer1_w,
                                            fc_layer1_bias,
                                            is_training,
                                            name="fc_layer1",
                                            drop_prob=fc_dropout_rate,
                                            activate_function=activate_function)

            fc_layer2_w = nn.init_weights(param["fc_layer2_w_shape"],
                                          initializer=initializer,
                                          name="fc_layer2/weight")
            fc_layer2_bias = nn.init_bias(param["fc_layer2_bias_shape"], name="fc_layer2/bias")
            fc_layer2 = nn.layer_perceptron(fc_layer1,
                                            fc_layer2_w,
                                            fc_layer2_bias,
                                            is_training,
                                            name="fc_layer2",
                                            drop_prob=fc_dropout_rate,
                                            activate_function=activate_function)

            # fc_layer3 = nn.layer_perceptron(fc_layer2,
            #                                 [layer_width],
            #                                 [layer_width],
            #                                 "fc_layer3",
            #                                 drop_prob=fc_dropout_rate,
            #                                 activate_function=activate_function)
            #
            # fc_layer4 = nn.layer_perceptron(fc_layer3,
            #                                 [layer_width],
            #                                 [layer_width],
            #                                 "fc_layer4",
            #                                 drop_prob=fc_dropout_rate,
            #                                 activate_function=activate_function)
            #
            # fc_layer5 = nn.layer_perceptron(fc_layer4,
            #                                 [layer_width],
            #                                 [layer_width],
            #                                 "fc_layer5",
            #                                 drop_prob=fc_dropout_rate,
            #                                 activate_function=activate_function)
            #
            # fc_layer6 = nn.layer_perceptron(fc_layer5,
            #                                 [layer_width],
            #                                 [layer_width],
            #                                 "fc_layer6",
            #                                 drop_prob=fc_dropout_rate,
            #                                 activate_function=activate_function)
            #
            # fc_layer7 = nn.layer_perceptron(fc_layer6,
            #                                 [layer_width],
            #                                 [layer_width],
            #                                 "fc_layer7",
            #                                 drop_prob=fc_dropout_rate,
            #                                 activate_function=activate_function)
            #
            # fc_layer8 = nn.layer_perceptron(fc_layer7,
            #                                 [layer_width],
            #                                 [layer_width],
            #                                 "fc_layer8",
            #                                 drop_prob=fc_dropout_rate,
            #                                 activate_function=activate_function)
            #
            # fc_layer9 = nn.layer_perceptron(fc_layer8,
            #                                 [layer_width],
            #                                 [layer_width],
            #                                 "fc_layer9",
            #                                 drop_prob=fc_dropout_rate,
            #                                 activate_function=activate_function)
            #
            # fc_layer10 = nn.layer_perceptron(fc_layer9,
            #                                  [layer_width],
            #                                  [layer_width],
            #                                  "fc_layer10",
            #                                  activate_function=activate_function)

        # softmax
        softmax_w = nn.init_weights(param["softmax_w_shape"])
        h = tf.nn.softmax(tf.matmul(fc_layer2, softmax_w))

        # print(h)
        # cross entropy
        # cost = tf.reduce_mean(-tf.reduce_sum(ph_set["Y"] * tf.log(h), reduction_indices=1))
        cost = -tf.reduce_sum(ph_set["Y"] * tf.log(h + 1e-10))
        l2_regularizer = tf.reduce_mean(tf.nn.l2_loss(conv1_w)
                                        + tf.nn.l2_loss(conv2_w)
                                        + tf.nn.l2_loss(conv3_w)
                                        + tf.nn.l2_loss(fc_layer1_w)
                                        + tf.nn.l2_loss(fc_layer2_w)
                                        + tf.nn.l2_loss(softmax_w))

        beta = param["l2_regularizer_beta"]
        cost = tf.add(cost, l2_regularizer * beta)

        # train_op
        learning_rate = param["learning_rate"]
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        # train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)
        # train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

        # prediction and acc
        predicted_label = tf.equal(tf.argmax(h, 1), tf.argmax(ph_set["Y"], 1))
        batch_acc = tf.reduce_mean(tf.cast(predicted_label, tf.float32))

        init_op = tf.global_variables_initializer()

        summary = tf.summary.merge_all()
        # save tensor
        tensor_set = {"X": ph_set["X"],
                      "Y": ph_set["Y"],
                      "Y_label": ph_set["Y_label"],
                      # "layer1": layer1,
                      # "layer2": layer2,
                      # "layer3": layer3,
                      # "W_softmax ": W_softmax,
                      # "h": h,
                      "cost": cost,
                      "train_op": train_op,
                      "predicted_label": predicted_label,
                      "batch_acc": batch_acc,
                      # "batch_hit_count ": batch_hit_count,
                      "init_op": init_op,
                      "summary": summary,
                      "fc_dropout_rate": fc_dropout_rate,
                      "conv_dropout_rate": conv_dropout_rate,
                      "is_training": is_training,
                      "global_epoch": global_epoch,
                      "inc_global_epoch": inc_global_epoch
                      }
        return tensor_set

    def gen_param_random(self):
        param = {}
        two_list = [2**i for i in range(0,12+1)]
        param["epoch_size"] = 200
        param["train_size"] = 5000

        param["mini_batch_size"] = two_list[random.randint(3, 10)]
        param["conv_dropout_rate"] = 0.7
        param["fc_dropout_rate"] = 0.5
        param["initializer"] = tf.contrib.layers.xavier_initializer

        param["conv1_out"] = two_list[random.randint(4, 9)]
        param["conv1_filter_size"] = random.randint(3, 5)
        param["conv1_w_shape"] = [param["conv1_filter_size"],
                                  param["conv1_filter_size"],
                                  3,
                                  param["conv1_out"]]

        param["conv2_out"] = two_list[random.randint(4, 9)]
        param["conv2_filter_size"] = random.randint(3, 5)
        param["conv2_w_shape"] = [param["conv2_filter_size"],
                                  param["conv2_filter_size"],
                                  param["conv1_out"],
                                  param["conv2_out"]]

        param["conv3_out"] = two_list[random.randint(4, 9)]
        param["conv3_filter_size"] = random.randint(3, 5)
        param["conv3_w_shape"] = [param["conv3_filter_size"],
                                  param["conv3_filter_size"],
                                  param["conv2_out"],
                                  param["conv3_out"]]

        param["fc_layer_depth"] = 2
        param["fc_layer_width"] = two_list[random.randint(4, 12)]
        param["fc_input_size"] = 4 * 4 * param["conv3_out"]
        param["fc_layer1_w_shape"] = [4 * 4 * param["conv3_out"], param["fc_layer_width"]]
        param["fc_layer1_bias_shape"] = [param["fc_layer_width"]]
        param["fc_layer2_w_shape"] = [param["fc_layer_width"], param["fc_layer_width"]]
        param["fc_layer2_bias_shape"] = [param["fc_layer_width"]]
        param["softmax_w_shape"] = [param["fc_layer_width"], 10]
        param["l2_regularizer_beta"] = random.uniform(0, 1)
        param["learning_rate"] = 10 ** (random.randint(-5, -4))

        param["param_list"] = self.PARAM_LIST
        return param

    pass
