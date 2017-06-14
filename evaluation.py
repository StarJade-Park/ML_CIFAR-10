import tensorflow as tf
import time
import datetime
import os

import util
import model as Model
from model import Model_cnn_nn_softmax_A
from model import Model_cnn_nn_softmax_B

import Batch as B
from Batch import Batch
from Batch import Config

label_name_list = [b'airplane',
                   b'automobile',
                   b'bird',
                   b'cat',
                   b'deer',
                   b'dog',
                   b'frog',
                   b'horse',
                   b'ship',
                   b'truck', ]


def model_evaluation(path, evaluation_model, evaluation_batch):
    param = util.load_param(path)
    model = evaluation_model.build_model(param)

    with tf.Session() as sess:
        sess.run(model[Model.INIT_OP])

        # restore session
        saver_path = os.path.join(path, "check_point")
        saver = tf.train.Saver()
        saver.restore(sess, saver_path)

        # calculate label_cnt, label_hit_cnt
        label_cnt = [0] * 10
        label_hit_cnt = [0] * 10
        mini_batch_size = 1000
        key_list = [B.INPUT_DATA, B.OUTPUT_LABEL, B.OUTPUT_DATA]
        for step in range(int(evaluation_batch.batch_size / mini_batch_size)):
            data = evaluation_batch.next_batch(mini_batch_size, key_list)
            feed_dict = {model["X"]: data[B.INPUT_DATA],
                         model["Y"]: data[B.OUTPUT_DATA],
                         model["Y_label"]: data[B.OUTPUT_LABEL],
                         model[Model.CONV_DROPOUT_RATE]: 1,
                         model[Model.FC_DROPOUT_RATE]: 1,
                         model[Model.IS_TRAINING]: False,
                         }
            predicted_label, h = sess.run([model[Model.PREDICTED_LABEL],
                                           model[Model.HYPOTHESIS]],
                                          feed_dict=feed_dict)

            for i in range(mini_batch_size):
                label = data[B.OUTPUT_LABEL][i]
                predicted_label_, h_ = predicted_label[i], h[i]

                label_cnt[label] += 1
                if label == predicted_label_:
                    label_hit_cnt[label] += 1

        # print evaluation result
        print("label name     | acc")
        print("------------------------------------")
        for i in range(10):
            acc = label_hit_cnt[i] / label_cnt[i]
            label_name = str(label_name_list[i])[2:-1]
            s = label_name + " " * (14 - len(label_name))
            print("%s | acc %.4f" % (s, acc))
        print("------------------------------------")
        total_acc = sum(label_hit_cnt) / sum(label_cnt)
        print("total acc : %.4f " % total_acc)
        print()

        # this make clear all graphs
    tf.reset_default_graph()


if __name__ == '__main__':
    util.pre_load()
    test_batch_config = Config(Config.OPTION_TEST_SET)
    test_batch = Batch(test_batch_config)
    train_batch_config = Config(Config.OPTION_TRAIN_SET, 1, 5)
    train_batch = Batch(train_batch_config)

    path = ".\\save\\tuning\\2017-06-14_09_20_19.395996"
    # path = ".\\save\\model_A backUp\\2017-06-13_13_37_44.075433"

    print("test batch evaluation")
    model_evaluation(path, Model_cnn_nn_softmax_A(), test_batch)

    print("train batch evaluation")
    model_evaluation(path, Model_cnn_nn_softmax_A(), train_batch)

    pass
