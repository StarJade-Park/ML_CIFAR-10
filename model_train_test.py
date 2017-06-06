import tensorflow as tf
import datetime
import os

import Batch
import all_flags
import tensor_summary as ts
import neural_networks as nn

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


def model_CNN_NN_softmax():
    ph_set = nn.placeholders_init()

    x_ = tf.reshape(ph_set["X"], [-1, 32, 32, 3])

    global_step = tf.Variable(initial_value=0,
                              name='global_step', trainable=False)

    conv_dropout_rate = tf.placeholder(tf.float32, name="conv_dropout_rate")
    fc_dropout_rate = tf.placeholder(tf.float32, name="fc_dropout_rate")

    # conv layer 1,2,3

    shape = [5, 5, 3, 64]
    conv1 = nn.conv2d_layer(x_,
                            shape,
                            keep_prob=conv_dropout_rate,
                            name="conv1")
    print(conv1)

    shape = [5, 5, 64, 64]
    conv2 = nn.conv2d_layer(conv1,
                            shape,
                            keep_prob=conv_dropout_rate,
                            name="conv2")
    print(conv2)

    pooling1 = nn.pooling2d_layer(conv2,
                                  keep_prob=conv_dropout_rate,
                                  name="pooling1")
    print(pooling1)

    shape = [5, 5, 64, 128]
    conv3 = nn.conv2d_layer(pooling1,
                            shape,
                            keep_prob=conv_dropout_rate,
                            name="conv3")
    print(conv3)

    shape = [5, 5, 128, 128]
    conv4 = nn.conv2d_layer(conv3,
                            shape,
                            keep_prob=conv_dropout_rate,
                            name="conv4")
    print(conv4)

    pooling2 = nn.pooling2d_layer(conv4,
                                  keep_prob=conv_dropout_rate,
                                  name="pooling2")
    print(pooling2)

    shape = [3, 3, 128, 256]
    conv5 = nn.conv2d_layer(pooling2,
                            shape,
                            keep_prob=conv_dropout_rate,
                            name="conv5")
    print(conv5)

    shape = [3, 3, 256, 256]
    conv6 = nn.conv2d_layer(conv5,
                            shape,
                            keep_prob=conv_dropout_rate,
                            name="conv6")
    print(conv6)

    pooling3 = nn.pooling2d_layer(conv6,
                                  keep_prob=conv_dropout_rate,
                                  name="pooling3")
    print(pooling3)
    # full connect layer

    fc_input_size = 4 * 4 * 256
    conv_out_reshape = tf.reshape(pooling3, [-1, fc_input_size])

    layer_width = 4096
    print(conv_out_reshape)
    activate_function = tf.nn.relu
    with tf.name_scope("fc_layer"):
        fc_layer1 = nn.layer_perceptron(conv_out_reshape,
                                        [fc_input_size],
                                        [layer_width],
                                        "fc_layer1",
                                        drop_prob=fc_dropout_rate,
                                        activate_function=activate_function)

        fc_layer2 = nn.layer_perceptron(fc_layer1,
                                        [layer_width],
                                        [layer_width],
                                        "fc_layer2",
                                        drop_prob=fc_dropout_rate,
                                        activate_function=activate_function)

        fc_layer3 = nn.layer_perceptron(fc_layer2,
                                        [layer_width],
                                        [layer_width],
                                        "fc_layer3",
                                        drop_prob=fc_dropout_rate,
                                        activate_function=activate_function)

        fc_layer4 = nn.layer_perceptron(fc_layer3,
                                        [layer_width],
                                        [layer_width],
                                        "fc_layer4",
                                        drop_prob=fc_dropout_rate,
                                        activate_function=activate_function)

        fc_layer5 = nn.layer_perceptron(fc_layer4,
                                        [layer_width],
                                        [layer_width],
                                        "fc_layer5",
                                        drop_prob=fc_dropout_rate,
                                        activate_function=activate_function)

        fc_layer6 = nn.layer_perceptron(fc_layer5,
                                        [layer_width],
                                        [layer_width],
                                        "fc_layer6",
                                        drop_prob=fc_dropout_rate,
                                        activate_function=activate_function)

        fc_layer7 = nn.layer_perceptron(fc_layer6,
                                        [layer_width],
                                        [layer_width],
                                        "fc_layer7",
                                        drop_prob=fc_dropout_rate,
                                        activate_function=activate_function)

        fc_layer8 = nn.layer_perceptron(fc_layer7,
                                        [layer_width],
                                        [layer_width],
                                        "fc_layer8",
                                        drop_prob=fc_dropout_rate,
                                        activate_function=activate_function)

        fc_layer9 = nn.layer_perceptron(fc_layer8,
                                        [layer_width],
                                        [layer_width],
                                        "fc_layer9",
                                        drop_prob=fc_dropout_rate,
                                        activate_function=activate_function)

        fc_layer10 = nn.layer_perceptron(fc_layer9,
                                         [layer_width],
                                         [layer_width],
                                         "fc_layer10",
                                         activate_function=activate_function)


    # softmax
    W_softmax = nn.init_weights([layer_width, 10])
    h = tf.nn.softmax(tf.matmul(fc_layer2, W_softmax))

    print(h)
    # cross entropy
    # cost = tf.reduce_mean(-tf.reduce_sum(ph_set["Y"] * tf.log(h), reduction_indices=1))
    cost = -tf.reduce_sum(ph_set["Y"] * tf.log(h + 1e-10))

    # train_op
    learning_rate = 1e-4
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
                  }
    return tensor_set


# TODO split function train_model and test_model
def train_and_model(model):
    # TODO LOOK ME this make show all tensor belong cpu or gpu
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess, tf.device("/CPU:0"):
    with tf.Session() as sess:

        # checkpoint val
        saver = tf.train.Saver()

        # tensorboard
        train_writer = tf.summary.FileWriter(FLAGS.dir_train_tensorboard, sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.dir_test_tensorboard)

        # init batch
        train_batch_config = Batch.Config(Batch.Config.OPTION_TRAIN_SET)
        train_batch = Batch.Batch(train_batch_config)
        test_batch_config = Batch.Config(Batch.Config.OPTION_TEST_SET)
        test_batch = Batch.Batch(test_batch_config)

        key_list = [Batch.INPUT_DATA, Batch.OUTPUT_LABEL, Batch.OUTPUT_DATA]

        # train step
        print(datetime.datetime.utcnow(), "Train Start...")

        sess.run(model["init_op"])

        train_acc = 1
        test_acc = 2
        for epoch in range(1000):
            mini_batch_size = 100
            step_size = int(50000 / mini_batch_size)

            conv_dropout_rate = .7
            fc_dropout_rate = .5

            # # conv_dropout_rate = max(1 - epoch * 0.001, 0.7)
            # print(test_acc + 0.01, train_acc)
            # print(test_acc + 0.01 < train_acc)
            # if test_acc + 1.0 < train_acc:
            #     fc_dropout_rate = 0.7
            # else:
            #     fc_dropout_rate = 1
            # print("fc_dropout_rate :", fc_dropout_rate)

            train_acc = 0.
            train_cost_mean = 0.0
            for step in range(step_size):
                data = train_batch.next_batch(mini_batch_size, key_list)
                feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                             model["Y"]: data[Batch.OUTPUT_DATA],
                             model["Y_label"]: data[Batch.OUTPUT_LABEL],
                             model["conv_dropout_rate"]: conv_dropout_rate,
                             model["fc_dropout_rate"]: fc_dropout_rate}

                sess.run(model["train_op"], feed_dict)
                # print log
                _acc, _cost = sess.run([model["batch_acc"], model["cost"]],
                                       feed_dict=feed_dict)
                train_acc += _acc / step_size
                train_cost_mean += _cost / step_size
                # print(step)

            print(datetime.datetime.utcnow(),
                  "epoch: %d" % epoch,
                  "train_acc: %f" % train_acc,
                  "cost: %f" % train_cost_mean)

            # test step
            mini_batch_size = 1000
            step_size = int(10000 / mini_batch_size)

            test_acc = 0.
            for step in range(step_size):
                data = test_batch.next_batch(mini_batch_size, key_list)
                feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                             model["Y"]: data[Batch.OUTPUT_DATA],
                             model["Y_label"]: data[Batch.OUTPUT_LABEL],
                             model["conv_dropout_rate"]: 1,
                             model["fc_dropout_rate"]: 1}

                _acc = sess.run(model["batch_acc"], feed_dict=feed_dict)
                test_acc += _acc / step_size

            print(datetime.datetime.utcnow(),
                  "epoch: %d" % epoch,
                  "test acc = %f" % test_acc)

    # this make clear all graphs
    tf.reset_default_graph()

    return


if __name__ == '__main__':

    # dirs exist check & make dirs
    if not os.path.exists(FLAGS.dir_train_checkpoint):
        os.makedirs(FLAGS.dir_train_checkpoint)

    if not os.path.exists(FLAGS.dir_test_checkpoint):
        os.makedirs(FLAGS.dir_test_checkpoint)

    if not os.path.exists(FLAGS.dir_train_tensorboard):
        os.makedirs(FLAGS.dir_train_tensorboard)

    if not os.path.exists(FLAGS.dir_test_tensorboard):
        os.makedirs(FLAGS.dir_test_tensorboard)

    print("CNN_NN_softmax")
    train_and_model(model_CNN_NN_softmax())

    # print("Neural Networks")
    # train_and_model(model_NN())

    # print("NN softmax")
    # train_and_model(model_NN_softmax())



    pass
