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
import numpy as np

# import matplotlib.pyplot as plt
# import math

# import all_flags
# flags = tf.app.flags
# FLAGS = flags.FLAGS

train_batch_config = None
train_batch = None
test_batch_config = None
test_batch = None


# load train and test batch
def load_batch():
    global train_batch_config, train_batch, test_batch_config, test_batch

    train_batch_config = Batch.Config(Batch.Config.OPTION_TRAIN_SET, 1, 15)
    train_batch = Batch.Batch(train_batch_config)
    test_batch_config = Batch.Config(Batch.Config.OPTION_TEST_SET)
    test_batch = Batch.Batch(test_batch_config)


# train model with test
def train_with_test(model, param, folder_path, is_restored=False):
    test_acc_list = []
    train_acc_list = []
    train_cost_list = []
    test_cost_list = []

    key_list = [Batch.INPUT_DATA, Batch.OUTPUT_LABEL, Batch.OUTPUT_DATA]

    model = model.build_model(param)
    print("print build model")

    saver_path = os.path.join(folder_path, "check_point")
    saver = tf.train.Saver()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(folder_path, sess.graph)

        sess.run(model["init_op"])

        # restore session
        if is_restored:
            saver.restore(sess, saver_path)
            print("session restored")

        print(datetime.datetime.utcnow(), "Train Start...")
        print("step        | time     | train acc | test acc  | train cost   | test cost")

        for i_ in range(param[Model.CHECK_POINT_INTERVAL]):
            train_acc = 0.
            train_cost_mean = 0.0
            test_acc = 0.
            test_cost_mean = 0.0
            start = time.time()

            # train step
            step_size = int(param[Model.TRAIN_SIZE] / param[Model.MINI_BATCH_SIZE])
            for step in range(step_size):
                data = train_batch.next_batch(param[Model.MINI_BATCH_SIZE], key_list)
                feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                             model["Y"]: data[Batch.OUTPUT_DATA],
                             model["Y_label"]: data[Batch.OUTPUT_LABEL],
                             model[Model.CONV_DROPOUT_RATE]: param["conv_dropout_rate"],
                             model[Model.FC_DROPOUT_RATE]: param["fc_dropout_rate"],
                             model[Model.IS_TRAINING]: True,
                             }

                _, _acc, _cost = sess.run([model[Model.TRAIN_OP],
                                           model[Model.BATCH_ACC],
                                           model[Model.L2_COST]],
                                          feed_dict=feed_dict)

                train_acc += (_acc / step_size) * 100
                train_cost_mean += _cost / step_size

            # test step
            mini_batch_size = param[Model.MINI_BATCH_SIZE]
            step_size = int(param[Model.TEST_SIZE] / param[Model.MINI_BATCH_SIZE])
            for step in range(step_size):
                data = test_batch.next_batch(mini_batch_size, key_list)
                feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                             model["Y"]: data[Batch.OUTPUT_DATA],
                             model["Y_label"]: data[Batch.OUTPUT_LABEL],
                             model[Model.CONV_DROPOUT_RATE]: 1,
                             model[Model.FC_DROPOUT_RATE]: 1,
                             model[Model.IS_TRAINING]: False,
                             }

                _acc, _cost = sess.run([model[Model.BATCH_ACC],
                                        model[Model.COST]],
                                       feed_dict=feed_dict)

                test_acc += (_acc / step_size) * 100
                test_cost_mean += _cost / step_size

            global_epoch = sess.run(model[Model.GLOBAL_STEP])
            # sess.run(model[Model.INC_GLOBAL_STEP])

            print("%2.d      |" % global_epoch,
                  "%.2f(s) |" % (time.time() - start),
                  "%.4f  |" % train_acc,
                  "%.4f  |" % test_acc,
                  "%f   |" % train_cost_mean,
                  "%f   |" % test_cost_mean
                  )

            train_acc_list += [train_acc]
            test_acc_list += [test_acc]
            train_cost_list += [train_cost_mean]
            test_cost_list += [test_cost_mean]

        saver.save(sess, saver_path)
        print("check point saved")

    # this make clear all graphs
    tf.reset_default_graph()
    print("clean up")

    return global_epoch, list(zip(train_acc_list,
                                  test_acc_list,
                                  train_cost_list,
                                  test_cost_list))


def start_with_new_model(train_model):
    param = train_model.default_param()
    util.print_param(param)

    folder_path = util.get_new_tuning_folder()

    epoch, output = train_with_test(train_model, param, folder_path)

    # save output and param
    util.save_output(output, epoch, folder_path)
    util.save_param(param, folder_path)

    return folder_path


def continue_train_model(train_model, path):
    param = util.load_param(path)
    util.print_param(param)

    for i in range(100):
        epoch, output = train_with_test(train_model,
                                        param,
                                        path,
                                        is_restored=True)

        util.save_output(output, epoch, path)


if __name__ == '__main__':
    util.pre_load()
    load_batch()

    path = start_with_new_model(Model_cnn_nn_softmax_A())

    continue_train_model(Model_cnn_nn_softmax_A(), path)
    # path = ".\\save\\model_A backUp\\2017-06-13_13_37_44.075433"
    # util.pre_load()
    # load_batch()
    #
    # path = ".\\save\\tuning\\2017-06-13_21_20_41.103668"
    # path = ".\\save\\tuning\\2017-06-13_21_20_41.103668"
    # continue_train(Model_cnn_nn_softmax_A(), path)
    pass
