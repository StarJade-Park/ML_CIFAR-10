import tensorflow as tf
import datetime
import os
import time
import tensor_summary as ts

import Batch
from model import Model_cnn_nn_softmax_B
from model import Model_cnn_nn_softmax_A
import model as Model

import util
import cProfile
import re

# import all_flags
# flags = tf.app.flags
# FLAGS = flags.FLAGS

train_batch_config = None
train_batch = None
test_batch_config = None
test_batch = None


def load_batch():
    global train_batch_config, train_batch, test_batch_config, test_batch

    train_batch_config = Batch.Config(Batch.Config.OPTION_TRAIN_SET)
    train_batch = Batch.Batch(train_batch_config)
    test_batch_config = Batch.Config(Batch.Config.OPTION_TEST_SET)
    test_batch = Batch.Batch(test_batch_config)

    train_batch.resume_thread()
    test_batch.resume_thread()

    print("batch_loaded start Q_ing....")


# TODO split function train_model and test_model
def train_and_test(model, param, folder_path, is_restored=False):
    test_acc_list = []
    train_acc_list = []
    train_cost_list = []
    test_cost_list = []
    global_epoch = 0

    key_list = [Batch.INPUT_DATA, Batch.OUTPUT_LABEL, Batch.OUTPUT_DATA]

    model = model.build_model(param)
    print("print build model")

    saver_path = os.path.join(folder_path, "check_point")

    saver = tf.train.Saver()

    # TODO LOOK ME this make show all tensor belong cpu or gpu
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess, tf.device("/CPU:0"):
    with tf.Session() as sess:
        # tensorboard
        # train_writer = tf.summary.FileWriter(FLAGS.dir_train_tensorboard, sess.graph)
        # test_writer = tf.summary.FileWriter(FLAGS.dir_test_tensorboard)

        print(datetime.datetime.utcnow(), "Train Start...")
        print("epoch| time     | train acc | test acc  | train cost   | test cost")
        sess.run(model["init_op"])

        if is_restored:
            saver.restore(sess, saver_path)
            print("restored")

        for epoch in range(param["epoch_size"]):
            train_acc = 0.
            train_cost_mean = 0.0
            test_acc = 0.
            test_cost_mean = 0.0

            start = time.time()

            # train step
            step_size = int(param["train_size"] / param["mini_batch_size"])
            for step in range(step_size):
                data = train_batch.next_batch(param["mini_batch_size"], key_list)
                feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                             model["Y"]: data[Batch.OUTPUT_DATA],
                             model["Y_label"]: data[Batch.OUTPUT_LABEL],
                             model["conv_dropout_rate"]: param["conv_dropout_rate"],
                             model["fc_dropout_rate"]: param["fc_dropout_rate"],
                             model["is_training"]: True,
                             }

                sess.run(model["train_op"], feed_dict)

                _acc, _cost = sess.run([model["batch_acc"], model["cost_L2"]],
                                       feed_dict=feed_dict)

                train_acc += (_acc / step_size) * 100
                train_cost_mean += _cost / step_size

            # test step
            mini_batch_size = param["mini_batch_size"]
            step_size = int(param["test_size"] / mini_batch_size)
            for step in range(step_size):
                data = test_batch.next_batch(mini_batch_size, key_list)
                feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                             model["Y"]: data[Batch.OUTPUT_DATA],
                             model["Y_label"]: data[Batch.OUTPUT_LABEL],
                             model["conv_dropout_rate"]: 1,
                             model["fc_dropout_rate"]: 1,
                             model["is_training"]: False,
                             }

                _acc, _cost = sess.run([model["batch_acc"], model["cost"]],
                                       feed_dict=feed_dict)

                test_acc += (_acc / step_size) * 100
                test_cost_mean += _cost / step_size

            global_epoch = sess.run(model["global_epoch"])
            sess.run(model["inc_global_epoch"])

            print("%2.d  |" % global_epoch,
                  "%.2f(s) |" % (time.time() - start),
                  "%.4f  |" % train_acc,
                  "%.4f  |" % test_acc,
                  "%f   |" % train_cost_mean,
                  "%f   |" % test_cost_mean
                  )

            train_acc_list += [train_acc]
            test_acc_list += [test_acc]
            train_cost_list += [train_cost_mean / mini_batch_size]
            test_cost_list += [test_cost_mean / mini_batch_size]

        saver.save(sess, saver_path)
        print("check point saved")

    # this make clear all graphs
    tf.reset_default_graph()
    print("clean up")

    return global_epoch, list(zip(train_acc_list,
                                  test_acc_list,
                                  train_cost_list,
                                  test_cost_list))


def gen_tuning_model(tuning_model, param, path):
    # print log
    print("gen tuning model")
    util.print_param(param)

    # gen folder and start training
    epoch, output = train_and_test(tuning_model, param, path)

    # save output and param
    util.save_output(output, epoch, path)
    util.save_param(param, path)


def resume_all_tuning_model():
    # TODO implement here

    for tuning_folder_path in util.get_all_tuning_folder_path():
        param = util.load_param(tuning_folder_path)
        restore_tuning_model()

    pass


def restore_tuning_model(tuning_model, param, path):
    # print log
    print("tuning folder path", path)
    # util.print_last_output(path)
    util.print_param(param)

    epoch, output = train_and_test(tuning_model,
                                   param,
                                   path,
                                   is_restored=True)
    util.save_output(output, epoch, path)


if __name__ == '__main__':
    # try:
    util.pre_load()
    load_batch()

    start = time.time()

    model = Model_cnn_nn_softmax_A()
    print("load model")

    param = model.default_param()
    # param[Model.CONV_DROPOUT_RATE] = 1
    # param[Model.FC_DROPOUT_RATE] = 1
    # param[Model.LEARNING_RATE] = 0.1
    print("get param")

    folder_path = util.get_new_tuning_folder()
    print("get folder path")

    gen_tuning_model(model, param, folder_path)

    # util.slack_bot("small epoch end")
    # msg = "time %d(s)" % (time.time() - start)
    # util.slack_bot(msg)
    # msg = util.print_last_output(folder_path)
    # util.slack_bot(msg)
    # msg = util.print_param(param)
    # util.slack_bot(msg)

    # gen new model
    print(util.get_all_tuning_folder_path())
    tuning_folder_path = folder_path
    for i in range(100):
        print(tuning_folder_path, i)
        start = time.time()
        param = util.load_param(tuning_folder_path)

        restore_tuning_model(model, param, tuning_folder_path)

        # if i %10 == 0:
        #     util.slack_bot("small epoch end")
        #     msg = "time %d(s)" % (time.time() - start)
        #     util.slack_bot(msg)
        #     msg = util.print_last_output(tuning_folder_path)
        #     util.slack_bot(msg)
        #     msg = util.print_param(param)
        #     util.slack_bot(msg)

        # except:
        #     # util.slack_bot("error look at me")
        #     pass
        # finally:
        #     train_batch.thread_exit()
        #     test_batch.thread_exit()
        #     # util.slack_bot("###### end")
        # pass
