import tensorflow as tf
import tensor_summary as ts
import neural_networks as nn
import random

# TODO write more comment please


flags = tf.app.flags
FLAGS = flags.FLAGS

PARAM_LIST = "param_list"
EPOCH_SIZE = "epoch_size"
TRAIN_SIZE = "train_size"
MINI_BATCH_SIZE = "mini_batch_size"
TEST_SIZE = "test_size"
INPUT_IMG_SIZE = "img_size"
INITIALIZER = "initializer"
FC_DROPOUT_RATE = "fc_dropout_rate"
CONV_DROPOUT_RATE = "conv_dropout_rate"

FORMAT_CONV_OUTPUT_SIZE = "conv%d_output_size"
FORMAT_CONV_INPUT_SIZE = "conv%d_input_size"
FORMAT_CONV_FILTER_SIZE = "conv%d_filter_size"
FORMAT_CONV_WEIGHT_SHAPE = "conv%d_weight_shape"

FC_INPUT_RESHAPE_SIZE = "fc_input_size"

FORMAT_FC_INPUT_SIZE = "format_fc%d_input_size"
FORMAT_FC_OUTPUT_SIZE = "format_fc%d_output_size"
FORMAT_FC_WEIGHT_SHAPE = "format_fc%d_weight_shape"
FORMAT_FC_BIAS_SHAPE = "format_fc%d_bias_shape"
FC_LAYER_DEPTH = "fc_layer_depth"

LEARNING_RATE = "learning_rate"
L2_REGULARIZER_BETA = "l2_regularizer_beta"
SOFTMAX_WEIGHT_SHAPE = "softmax_w_shape"


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
        batch_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_label, ph_set["Y_label"]), tf.float64),
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


class Model_cnn_nn_softmax_A:
    PARAM_LIST = [EPOCH_SIZE,
                  TRAIN_SIZE,
                  TEST_SIZE,
                  MINI_BATCH_SIZE,

                  CONV_DROPOUT_RATE,
                  FC_DROPOUT_RATE,

                  INITIALIZER,

                  INPUT_IMG_SIZE,

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
        param = dict()
        param["param_list"] = self.PARAM_LIST

        param[EPOCH_SIZE] = 25
        param[MINI_BATCH_SIZE] = 100
        param[TRAIN_SIZE] = 5000
        param[TEST_SIZE] = 1000

        #
        # param[EPOCH_SIZE] = 2
        # param[MINI_BATCH_SIZE] = 2
        # param[TRAIN_SIZE] = 4
        # param[TEST_SIZE] = 4

        # param[CONV_DROPOUT_RATE] = 0.25
        # param[FC_DROPOUT_RATE] = 0.25
        param[CONV_DROPOUT_RATE] = 1
        param[FC_DROPOUT_RATE] = 1

        param[INITIALIZER] = tf.contrib.layers.xavier_initializer

        param[INPUT_IMG_SIZE] = 32

        param["conv1_out"] = 128
        param["conv1_filter_size"] = 5
        param["conv1_w_shape"] = [param["conv1_filter_size"],
                                  param["conv1_filter_size"],
                                  3,
                                  param["conv1_out"]]

        param["conv2_out"] = 512
        param["conv2_filter_size"] = 5
        param["conv2_w_shape"] = [param["conv2_filter_size"],
                                  param["conv2_filter_size"],
                                  param["conv1_out"],
                                  param["conv2_out"]]

        param["conv3_out"] = 512
        param["conv3_filter_size"] = 3
        param["conv3_w_shape"] = [param["conv3_filter_size"],
                                  param["conv3_filter_size"],
                                  param["conv2_out"],
                                  param["conv3_out"]]

        param["fc_layer_depth"] = 2
        param["fc_layer_width"] = 512

        size = int(param[INPUT_IMG_SIZE] / 8)
        param["fc_input_size"] = size * size * param["conv3_out"]

        param["fc_layer1_w_shape"] = [param["fc_input_size"], param["fc_layer_width"]]
        param["fc_layer1_bias_shape"] = [param["fc_layer_width"]]
        param["fc_layer2_w_shape"] = [param["fc_layer_width"], param["fc_layer_width"]]
        param["fc_layer2_bias_shape"] = [param["fc_layer_width"]]

        param["softmax_w_shape"] = [param["fc_layer_width"], 10]
        param["l2_regularizer_beta"] = 0.003
        param["learning_rate"] = 1e-4

        return param

    def build_model(self, param):
        ph_set = nn.placeholders_init(param[INPUT_IMG_SIZE])

        x_ = tf.reshape(ph_set["X"], [-1, param[INPUT_IMG_SIZE], param[INPUT_IMG_SIZE], 3])
        is_training = tf.placeholder(tf.bool, name="is_training")

        global_step = tf.Variable(initial_value=0,
                                  name='global_step', trainable=False)

        global_epoch = tf.Variable(initial_value=0,
                                   name='global_epoch', trainable=False)

        inc_global_epoch = tf.assign(global_epoch, global_epoch + 1)

        conv_dropout_rate = tf.placeholder(tf.float32, name=CONV_DROPOUT_RATE)
        fc_dropout_rate = tf.placeholder(tf.float32, name=FC_DROPOUT_RATE)

        initializer = param[INITIALIZER]

        with tf.name_scope("conv_layer1"):
            conv1_w = nn.init_weights(param["conv1_w_shape"], initializer, "conv1/w")
            conv1 = nn.conv2d_layer(x_,
                                    conv1_w,
                                    is_training,
                                    keep_prob=conv_dropout_rate,
                                    name="conv1")

            pooling1 = nn.pooling2d_layer(conv1, name="pooling1")
        with tf.name_scope("conv_layer2"):
            conv2_w = nn.init_weights(param["conv2_w_shape"], initializer, "conv2/w")
            conv2 = nn.conv2d_layer(pooling1,
                                    conv2_w,
                                    is_training,
                                    keep_prob=conv_dropout_rate,
                                    name="conv2")

            pooling2 = nn.pooling2d_layer(conv2, name="pooling2")
        with tf.name_scope("conv_layer3"):
            conv3_w = nn.init_weights(param["conv3_w_shape"], initializer, "conv3/w")
            conv3 = nn.conv2d_layer(pooling2,
                                    conv3_w,
                                    is_training,
                                    keep_prob=conv_dropout_rate,
                                    name="conv3")

            pooling3 = nn.pooling2d_layer(conv3, name="pooling3")

        fc_input_size = param[FC_INPUT_RESHAPE_SIZE]
        conv_out_reshape = tf.reshape(pooling3, [-1, fc_input_size])

        # print(conv_out_reshape)
        activate_function = "relu"
        with tf.name_scope("fc_layer1"):
            fc_layer1_w = nn.init_weights(param["fc_layer1_w_shape"],
                                          initializer=initializer,
                                          decay=0.9999,
                                          name="fc_layer1/weight")

            fc_layer1_bias = nn.init_bias(param["fc_layer1_bias_shape"], name="fc_layer1/bias")

            fc_layer1 = nn.layer_perceptron(conv_out_reshape,
                                            fc_layer1_w,
                                            fc_layer1_bias,
                                            is_training,
                                            name="fc_layer1",
                                            drop_prob=fc_dropout_rate,
                                            activate_function=activate_function)

        with tf.name_scope("fc_layer2"):
            fc_layer2_w = nn.init_weights(param["fc_layer2_w_shape"],
                                          initializer=initializer,
                                          decay=0.9999,
                                          name="fc_layer2/weight")
            fc_layer2_bias = nn.init_bias(param["fc_layer2_bias_shape"], name="fc_layer2/bias")
            fc_layer2 = nn.layer_perceptron(fc_layer1,
                                            fc_layer2_w,
                                            fc_layer2_bias,
                                            is_training,
                                            name="fc_layer2",
                                            drop_prob=1,
                                            activate_function=activate_function)

        # softmax
        softmax_w = nn.init_weights(param[SOFTMAX_WEIGHT_SHAPE])
        h = tf.nn.softmax(tf.matmul(fc_layer2, softmax_w))
        # print(h)
        # cross entropy
        cost = tf.reduce_mean(-tf.reduce_sum(ph_set["Y"] * tf.log(h + 1e-10), reduction_indices=1), name="cost")
        # cost = -tf.reduce_sum(ph_set["Y"] * tf.log(h + 1e-10), name="cost")

        l2_regularizer = tf.reduce_mean(tf.nn.l2_loss(conv1_w)
                                        + tf.nn.l2_loss(conv2_w)
                                        + tf.nn.l2_loss(conv3_w)
                                        + tf.nn.l2_loss(fc_layer1_w)
                                        + tf.nn.l2_loss(fc_layer2_w)
                                        + tf.nn.l2_loss(softmax_w))

        beta = param[L2_REGULARIZER_BETA]
        cost_L2 = tf.add(cost, l2_regularizer * beta, name="cost_L2")

        # train_op
        learning_rate = param[LEARNING_RATE]
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_L2, global_step=global_step)
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
        # train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)
        # train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

        # prediction and acc

        predicted_label = tf.equal(tf.argmax(h, 1), tf.argmax(ph_set["Y"], 1))
        batch_acc = tf.reduce_mean(tf.cast(predicted_label, tf.float32))

        init_op = tf.global_variables_initializer()

        summary = tf.summary.merge_all()
        tensor_set = {"X": ph_set["X"],
                      "Y": ph_set["Y"],
                      "Y_label": ph_set["Y_label"],

                      "conv1": conv1,
                      "conv2": conv2,
                      "conv3": conv3,

                      "conv1_w": conv1_w,
                      "conv2_w": conv2_w,
                      "conv3_w": conv3_w,

                      "pooling1": pooling1,
                      "pooling2": pooling2,
                      "pooling3": pooling3,
                      "conv_out_reshape": conv_out_reshape,

                      "fc_layer1": fc_layer1,
                      "fc_layer2": fc_layer2,
                      "softmax_w": softmax_w,
                      "h": h,
                      "cost": cost,
                      "cost_L2": cost_L2,
                      "train_op": train_op,
                      "predicted_label": predicted_label,
                      "batch_acc": batch_acc,

                      # "batch_hit_count ": batch_hit_count,
                      "init_op": init_op,
                      "summary": summary,
                      FC_DROPOUT_RATE: fc_dropout_rate,
                      CONV_DROPOUT_RATE: conv_dropout_rate,
                      "is_training": is_training,
                      "global_epoch": global_epoch,
                      "inc_global_epoch": inc_global_epoch
                      }
        # save tensor
        return tensor_set

    def gen_param_random(self):
        param = {}
        two_list = [2 ** i for i in range(0, 12 + 1)]
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
        param["l2_regularizer_beta"] = random.uniform(0, 5)
        param["learning_rate"] = 10 ** (random.randint(-4, -4))

        param["param_list"] = self.PARAM_LIST
        return param


class Model_cnn_nn_softmax_B:
    PARAM_LIST = [EPOCH_SIZE,
                  TRAIN_SIZE,
                  TEST_SIZE,
                  MINI_BATCH_SIZE,

                  CONV_DROPOUT_RATE,
                  FC_DROPOUT_RATE,

                  INITIALIZER,

                  INPUT_IMG_SIZE,

                  "conv1_out",
                  "conv1_filter_size",
                  "conv1_w_shape",

                  "conv2_out",
                  "conv2_filter_size",
                  "conv2_w_shape",
                  "conv3_out",
                  "conv3_filter_size",
                  "conv3_w_shape",
                  "conv4_out",
                  "conv4_filter_size",
                  "conv4_w_shape",
                  "conv5_out",
                  "conv5_filter_size",
                  "conv5_w_shape",
                  "conv6_out",
                  "conv6_filter_size",
                  "conv6_w_shape",
                  "conv7_out",
                  "conv7_filter_size",
                  "conv7_w_shape",
                  "conv8_out",
                  "conv8_filter_size",
                  "conv8_w_shape",
                  "conv9_out",
                  "conv9_filter_size",
                  "conv9_w_shape",

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
        param = dict()
        param[PARAM_LIST] = self.PARAM_LIST

        param[EPOCH_SIZE] = 20
        param[MINI_BATCH_SIZE] = 100
        param[TRAIN_SIZE] = 1000
        param[TEST_SIZE] = 1000
        #
        # param[EPOCH_SIZE] = 2
        # param[MINI_BATCH_SIZE] = 2
        # param[TRAIN_SIZE] = param[MINI_BATCH_SIZE] * 2
        # param[TEST_SIZE] = param[MINI_BATCH_SIZE] * 2

        param[INPUT_IMG_SIZE] = 24

        param[CONV_DROPOUT_RATE] = 0.7
        param[FC_DROPOUT_RATE] = 0.5
        param[INITIALIZER] = tf.contrib.layers.xavier_initializer

        param["conv1_out"] = 128
        param["conv1_filter_size"] = 3
        param["conv1_w_shape"] = [param["conv1_filter_size"],
                                  param["conv1_filter_size"],
                                  3,
                                  param["conv1_out"]]

        param["conv2_out"] = 128
        param["conv2_filter_size"] = 3
        param["conv2_w_shape"] = [param["conv2_filter_size"],
                                  param["conv2_filter_size"],
                                  param["conv1_out"],
                                  param["conv2_out"]]

        param["conv3_out"] = 128
        param["conv3_filter_size"] = 3
        param["conv3_w_shape"] = [param["conv3_filter_size"],
                                  param["conv3_filter_size"],
                                  param["conv2_out"],
                                  param["conv3_out"]]

        param["conv4_out"] = 256
        param["conv4_filter_size"] = 3
        param["conv4_w_shape"] = [param["conv4_filter_size"],
                                  param["conv4_filter_size"],
                                  param["conv3_out"],
                                  param["conv4_out"]]

        param["conv5_out"] = 256
        param["conv5_filter_size"] = 3
        param["conv5_w_shape"] = [param["conv5_filter_size"],
                                  param["conv5_filter_size"],
                                  param["conv4_out"],
                                  param["conv5_out"]]

        param["conv6_out"] = 256
        param["conv6_filter_size"] = 3
        param["conv6_w_shape"] = [param["conv6_filter_size"],
                                  param["conv6_filter_size"],
                                  param["conv5_out"],
                                  param["conv6_out"]]

        param["conv7_out"] = 256
        param["conv7_filter_size"] = 3
        param["conv7_w_shape"] = [param["conv7_filter_size"],
                                  param["conv7_filter_size"],
                                  param["conv6_out"],
                                  param["conv7_out"]]

        param["conv8_out"] = 256
        param["conv8_filter_size"] = 3
        param["conv8_w_shape"] = [param["conv8_filter_size"],
                                  param["conv8_filter_size"],
                                  param["conv7_out"],
                                  param["conv8_out"]]

        param["conv9_out"] = 256
        param["conv9_filter_size"] = 3
        param["conv9_w_shape"] = [param["conv9_filter_size"],
                                  param["conv9_filter_size"],
                                  param["conv8_out"],
                                  param["conv9_out"]]

        param[FC_LAYER_DEPTH] = 2
        param["fc_layer_width"] = 2048
        param["fc_input_size"] = 3 * 3 * param["conv9_out"]

        param["fc_layer1_w_shape"] = [param["fc_input_size"], param["fc_layer_width"]]
        param["fc_layer1_bias_shape"] = [param["fc_layer_width"]]

        param["fc_layer2_w_shape"] = [param["fc_layer_width"], param["fc_layer_width"]]
        param["fc_layer2_bias_shape"] = [param["fc_layer_width"]]

        param[SOFTMAX_WEIGHT_SHAPE] = [param["fc_layer_width"], 10]
        param[L2_REGULARIZER_BETA] = 0.1
        param[LEARNING_RATE] = 1e-4

        return param

    def build_model(self, param):
        ph_set = nn.placeholders_init()

        x_ = tf.reshape(ph_set["X"], [-1, param[INPUT_IMG_SIZE], param[INPUT_IMG_SIZE], 3])
        is_training = tf.placeholder(tf.bool, name="is_training")

        global_step = tf.Variable(initial_value=0,
                                  name='global_step', trainable=False)

        global_epoch = tf.Variable(initial_value=0,
                                   name='global_epoch', trainable=False)

        inc_global_epoch = tf.assign(global_epoch, global_epoch + 1)

        conv_dropout_rate = tf.placeholder(tf.float32, name=CONV_DROPOUT_RATE)
        fc_dropout_rate = tf.placeholder(tf.float32, name=FC_DROPOUT_RATE)

        initializer = param[INITIALIZER]

        with tf.name_scope("conv_layer"):
            conv1_w = nn.init_weights(param["conv1_w_shape"], initializer, "conv1/w")
            conv1 = nn.conv2d_layer(x_,
                                    conv1_w,
                                    is_training,
                                    keep_prob=conv_dropout_rate,
                                    name="conv1")
            conv2_w = nn.init_weights(param["conv2_w_shape"], initializer, "conv2/w")
            conv2 = nn.conv2d_layer(conv1,
                                    conv2_w,
                                    is_training,
                                    keep_prob=conv_dropout_rate,
                                    name="conv2")

            conv3_w = nn.init_weights(param["conv3_w_shape"], initializer, "conv3/w")
            conv3 = nn.conv2d_layer(conv2,
                                    conv3_w,
                                    is_training,
                                    keep_prob=conv_dropout_rate,
                                    name="conv3")

            pooling1 = nn.pooling2d_layer(conv3, name="pooling1")

            conv4_w = nn.init_weights(param["conv4_w_shape"], initializer, "conv4/w")
            conv4 = nn.conv2d_layer(pooling1,
                                    conv4_w,
                                    is_training,
                                    keep_prob=conv_dropout_rate,
                                    name="conv4")

            conv5_w = nn.init_weights(param["conv5_w_shape"], initializer, "conv5/w")
            conv5 = nn.conv2d_layer(conv4,
                                    conv5_w,
                                    is_training,
                                    keep_prob=conv_dropout_rate,
                                    name="conv5")

            conv6_w = nn.init_weights(param["conv6_w_shape"], initializer, "conv6/w")
            conv6 = nn.conv2d_layer(conv5,
                                    conv6_w,
                                    is_training,
                                    keep_prob=conv_dropout_rate,
                                    name="conv6")

            pooling2 = nn.pooling2d_layer(conv6, name="pooling2")

            conv7_w = nn.init_weights(param["conv7_w_shape"], initializer, "conv7/w")
            conv7 = nn.conv2d_layer(pooling2,
                                    conv7_w,
                                    is_training,
                                    keep_prob=conv_dropout_rate,
                                    name="conv7")

            conv8_w = nn.init_weights(param["conv8_w_shape"], initializer, "conv8/w")
            conv8 = nn.conv2d_layer(conv7,
                                    conv8_w,
                                    is_training,
                                    keep_prob=conv_dropout_rate,
                                    name="conv8")

            conv9_w = nn.init_weights(param["conv9_w_shape"], initializer, "conv9/w")
            conv9 = nn.conv2d_layer(conv8,
                                    conv9_w,
                                    is_training,
                                    keep_prob=conv_dropout_rate,
                                    name="conv9")

            pooling3 = nn.pooling2d_layer(conv9, name="pooling3")

        fc_input_size = param["fc_input_size"]
        conv_out_reshape = tf.reshape(pooling3, [-1, fc_input_size])

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
                                            drop_prob=1,
                                            activate_function=activate_function)

        # softmax
        softmax_w = nn.init_weights(param[SOFTMAX_WEIGHT_SHAPE])
        h = tf.nn.softmax(tf.matmul(fc_layer2, softmax_w))

        # print(h)
        # cross entropy
        cost = tf.reduce_mean(-tf.reduce_sum(ph_set["Y"] * tf.log(h + 1e-10), reduction_indices=1), name="cost")
        # cost = -tf.reduce_sum(ph_set["Y"] * tf.log(h + 1e-10), name="cost")

        l2_regularizer = tf.reduce_mean(tf.nn.l2_loss(conv1_w)
                                        + tf.nn.l2_loss(conv2_w)
                                        + tf.nn.l2_loss(conv3_w)
                                        + tf.nn.l2_loss(conv4_w)
                                        + tf.nn.l2_loss(conv5_w)
                                        + tf.nn.l2_loss(conv6_w)
                                        + tf.nn.l2_loss(conv7_w)
                                        + tf.nn.l2_loss(conv8_w)
                                        + tf.nn.l2_loss(conv9_w)
                                        + tf.nn.l2_loss(fc_layer1_w)
                                        + tf.nn.l2_loss(fc_layer2_w)
                                        + tf.nn.l2_loss(softmax_w))

        beta = param[L2_REGULARIZER_BETA]
        cost_L2 = tf.add(cost, l2_regularizer * beta, name="cost_L2")

        # train_op
        learning_rate = param[LEARNING_RATE]
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_L2, global_step=global_step, name="train_op")
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        # train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)
        # train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

        # prediction and acc
        predicted_label = tf.equal(tf.argmax(h, 1), tf.argmax(ph_set["Y"], 1))
        batch_acc = tf.reduce_mean(tf.cast(predicted_label, tf.float32))

        init_op = tf.global_variables_initializer()

        summary = tf.summary.merge_all()
        tensor_set = {"X": ph_set["X"],
                      "Y": ph_set["Y"],
                      "Y_label": ph_set["Y_label"],

                      "conv1": conv1,
                      "conv2": conv2,
                      "conv3": conv3,
                      "conv4": conv4,
                      "conv5": conv5,
                      "conv6": conv6,
                      "conv7": conv7,
                      "conv8": conv8,
                      "conv9": conv9,

                      "conv1_w": conv1_w,
                      "conv2_w": conv2_w,
                      "conv3_w": conv3_w,
                      "conv4_w": conv4_w,
                      "conv5_w": conv5_w,
                      "conv6_w": conv6_w,
                      "conv7_w": conv7_w,
                      "conv8_w": conv8_w,
                      "conv9_w": conv9_w,

                      "pooling1": pooling1,
                      "pooling2": pooling2,
                      "pooling3": pooling3,
                      "conv_out_reshape": conv_out_reshape,

                      "fc_layer1": fc_layer1,
                      "fc_layer2": fc_layer2,
                      "softmax_w": softmax_w,
                      "h": h,
                      "cost": cost,
                      "cost_L2": cost_L2,
                      "train_op": train_op,
                      "predicted_label": predicted_label,
                      "batch_acc": batch_acc,

                      # "batch_hit_count ": batch_hit_count,
                      "init_op": init_op,
                      "summary": summary,
                      FC_DROPOUT_RATE: fc_dropout_rate,
                      CONV_DROPOUT_RATE: conv_dropout_rate,
                      "is_training": is_training,
                      "global_epoch": global_epoch,
                      "inc_global_epoch": inc_global_epoch
                      }
        # save tensor
        return tensor_set

    def gen_param_random(self):
        param = dict()
        two_list = [2 ** i for i in range(0, 12 + 1)]
        param[EPOCH_SIZE] = 200
        param["train_size"] = 5000

        param[MINI_BATCH_SIZE] = two_list[random.randint(3, 10)]
        param[CONV_DROPOUT_RATE] = 0.7
        param[FC_DROPOUT_RATE] = 0.5
        param[INITIALIZER] = tf.contrib.layers.xavier_initializer

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
        param["l2_regularizer_beta"] = random.uniform(0, 5)
        param["learning_rate"] = 10 ** (random.randint(-4, -4))

        param["param_list"] = self.PARAM_LIST
        return param

    pass
