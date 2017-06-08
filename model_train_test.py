import tensorflow as tf
import datetime
import os

import Batch
import all_flags
import tensor_summary as ts
import neural_networks as nn
import random
import model
import util

# TODO file must split
# TODO write more comment please

flags = tf.app.flags
FLAGS = flags.FLAGS


# TODO split function train_model and test_model
def train_and_test(model, param, saver_path, is_restored=False):
    test_acc_list = []
    train_acc_list = []
    train_cost_list = []
    test_cost_list = []
    global_epoch = 0

    saver = tf.train.Saver()

    # TODO LOOK ME this make show all tensor belong cpu or gpu
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess, tf.device("/CPU:0"):
    with tf.Session() as sess:
        # checkpoint val

        # tensorboard
        # train_writer = tf.summary.FileWriter(FLAGS.dir_train_tensorboard, sess.graph)
        # test_writer = tf.summary.FileWriter(FLAGS.dir_test_tensorboard)

        # init batch
        train_batch_config = Batch.Config(Batch.Config.OPTION_TRAIN_SET)
        train_batch = Batch.Batch(train_batch_config)
        test_batch_config = Batch.Config(Batch.Config.OPTION_TEST_SET)
        test_batch = Batch.Batch(test_batch_config)

        key_list = [Batch.INPUT_DATA, Batch.OUTPUT_LABEL, Batch.OUTPUT_DATA]

        # train step
        print(datetime.datetime.utcnow(), "Train Start...")

        sess.run(model["init_op"])

        if is_restored:
            saver.restore(sess, saver_path)
            print("restored")

        for epoch in range(param["epoch_size"]):
            train_size = param["train_size"]
            mini_batch_size = param["mini_batch_size"]

            conv_dropout_rate = param["conv_dropout_rate"]
            fc_dropout_rate = param["fc_dropout_rate"]

            step_size = int(train_size / mini_batch_size)

            train_acc = 0.
            train_cost_mean = 0.0
            is_training = True
            for step in range(step_size):
                data = train_batch.next_batch(mini_batch_size, key_list, shuffle=True)
                feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                             model["Y"]: data[Batch.OUTPUT_DATA],
                             model["Y_label"]: data[Batch.OUTPUT_LABEL],
                             model["conv_dropout_rate"]: conv_dropout_rate,
                             model["fc_dropout_rate"]: fc_dropout_rate,
                             model["is_training"]: is_training,
                             }

                sess.run(model["train_op"], feed_dict)
                # print log
                _acc, _cost = sess.run([model["batch_acc"], model["cost"]],
                                       feed_dict=feed_dict)
                train_acc += _acc / step_size
                train_cost_mean += _cost / step_size

            # test step
            mini_batch_size = 1000
            step_size = int(10000 / mini_batch_size)

            test_acc = 0.
            test_cost_mean = 0.0
            is_training = False
            for step in range(step_size):
                data = test_batch.next_distorted_batch(mini_batch_size, key_list)
                feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                             model["Y"]: data[Batch.OUTPUT_DATA],
                             model["Y_label"]: data[Batch.OUTPUT_LABEL],
                             model["conv_dropout_rate"]: 1,
                             model["fc_dropout_rate"]: 1,
                             model["is_training"]: is_training,
                             }

                _acc, _cost = sess.run([model["batch_acc"], model["cost"]],
                                       feed_dict=feed_dict)
                test_acc += _acc / step_size
                test_cost_mean += _cost / step_size

            global_epoch = sess.run(model["global_epoch"])
            sess.run(model["inc_global_epoch"])
            print(datetime.datetime.utcnow(),
                  "epoch = %d" % global_epoch,
                  "train = %f" % train_acc,
                  "test = %f" % test_acc,
                  "cost = %f" % (train_cost_mean / mini_batch_size),
                  "test cost = %f" % (test_cost_mean / mini_batch_size)
                  )

            train_acc_list += [train_acc]
            test_acc_list += [test_acc]
            train_cost_list += [train_cost_mean / mini_batch_size]
            test_cost_list += [test_cost_mean / mini_batch_size]

        print("saved", saver.save(sess, saver_path))

    # this make clear all graphs
    tf.reset_default_graph()

    return global_epoch, zip(train_acc_list, test_acc_list, train_cost_list, test_cost_list)


if __name__ == '__main__':
    util.pre_load()

    # dirs exist check & make dirs
    # if not os.path.exists(FLAGS.dir_train_checkpoint):
    #     os.makedirs(FLAGS.dir_train_checkpoint)
    #
    # if not os.path.exists(FLAGS.dir_test_checkpoint):
    #     os.makedirs(FLAGS.dir_test_checkpoint)
    #
    # if not os.path.exists(FLAGS.dir_train_tensorboard):
    #     os.makedirs(FLAGS.dir_train_tensorboard)
    #
    # if not os.path.exists(FLAGS.dir_test_tensorboard):
    #     os.makedirs(FLAGS.dir_test_tensorboard)

    print("CNN_NN_softmax")

    cnn = model.Model_cnn_nn_softmax()

    for i in range(100):
        print("tunning %d" % i)

        param = cnn.gen_param_random()
        util.print_param(param)

        folder_name = str(datetime.datetime.utcnow()).replace(" ", "_").replace(":", "_")
        folder_path = os.path.join(".", "save", "tuning", folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        saver_file_name = "check_point"
        saver_path = os.path.join(folder_path, saver_file_name)

        param_file_name = "param"
        param_file_path = os.path.join(folder_path, param_file_name)
        util.pickle(param, param_file_path)
        print("save param")

        model = cnn.build_model(param)
        print("build model")

        epoch, out = train_and_test(model, param, saver_path)
        path = os.path.join(folder_path, "~epoch_" + str(epoch).zfill(6))
        util.pickle(out, path)
        print("save out")



    # param = util.unpickle(param_file_path)
    # print("restore param")
    #
    # model = cnn.build_model(param)
    #
    # print("build model")
    # out = train_and_test(model, param, saver_path, is_restored=True)
    # path = os.path.join(folder_path, "~epoch_" + str(epoch).zfill(6))
    # util.pickle(out, path)
    # print("save out")

    # print("Neural Networks")
    # train_and_model(model_NN())

    # print("NN softmax")
    # train_and_model(model_NN_softmax())



    pass
