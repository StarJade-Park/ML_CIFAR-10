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

    layer1 = nn.layer_perceptron(ph_set["X"], [FLAGS.image_size],
                                 [FLAGS.perceptron_input_shape_size], "layer_1")
    layer2 = nn.layer_perceptron(layer1, [FLAGS.perceptron_input_shape_size],
                                 [FLAGS.perceptron_output_shape_size], "layer_2")
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

        # train step
        print("Train Start...")
        train_batch_config = Batch.Config(Batch.Config.OPTION_TRAIN_SET)
        train_batch = Batch.Batch(train_batch_config)
        sess.run(model["init_op"])

        for step in range(FLAGS.max_train_step + 1):
            key_list = [Batch.INPUT_DATA, Batch.OUTPUT_LABEL, Batch.OUTPUT_DATA]

            data = train_batch.next_batch(FLAGS.batch_size, key_list)

            feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                         model["Y"]: data[Batch.OUTPUT_DATA],
                         model["Y_label"]: data[Batch.OUTPUT_LABEL]}

            sess.run(model["train_op"], feed_dict)
            # print log
            if step % FLAGS.print_log_step_size == 0:
                summary_train, _acc, _cost = sess.run([model["summary"], model["batch_acc"], model["cost"]],
                                                      feed_dict=feed_dict)
                print(datetime.datetime.utcnow(), "train step: %d" % step
                      , "batch_acc:", _acc, "cost:", _cost)

            # checkpoint
            if step % FLAGS.checkpoint_step_size == 0:
                saver.save(sess, FLAGS.dir_train_checkpoint, global_step=step)

            # summary tensorboard
            if step % FLAGS.summary_step_size:
                train_writer.add_summary(summary=summary_train, global_step=step)

        # test step
        print("Test Start...")
        test_batch_config = Batch.Config(Batch.Config.OPTION_TEST_SET)
        test_batch = Batch.Batch(test_batch_config)

        total_acc = 0.
        for step in range(FLAGS.max_test_step + 1):
            key_list = [Batch.INPUT_DATA, Batch.OUTPUT_LABEL, Batch.OUTPUT_DATA]

            data = test_batch.next_batch(FLAGS.batch_size, key_list)
            feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                         model["Y"]: data[Batch.OUTPUT_DATA],
                         model["Y_label"]: data[Batch.OUTPUT_LABEL]}

            summary_test, _acc = sess.run([model["summary"], model["batch_acc"]], feed_dict=feed_dict)
            print(datetime.datetime.utcnow(), "test step: %d" % step
                  , "batch_acc: ", _acc)
            total_acc += _acc

            if step % FLAGS.print_log_step_size == 0:
                summary_test, _acc = sess.run([model["summary"], model["batch_acc"]], feed_dict=feed_dict)
                # print(datetime.datetime.utcnow(), "test step: %d" % step
                #       , "batch_acc: ", _acc)

            test_writer.add_summary(summary=summary_test, global_step=step)

        print("test complete: total acc =", total_acc / (FLAGS.max_test_step + 1))

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

    print("Neural Networks")
    train_and_model(model_NN())

    print("NN softmax")
    train_and_model(model_NN_softmax())
    pass
