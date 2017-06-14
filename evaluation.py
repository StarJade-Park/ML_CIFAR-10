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
    print("evaluation model built")

    # TODO LOOK ME this make show all tensor belong cpu or gpu
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess, tf.device("/CPU:0"):
    with tf.Session() as sess:
        sess.run(model["init_op"])

        # restore session
        saver_path = os.path.join(path, "check_point")
        saver = tf.train.Saver()
        saver.restore(sess, saver_path)
        print("session restored")

        # evaluation
        mini_batch_size = 1
        step_size = evaluation_batch.batch_size
        key_list = [B.INPUT_DATA, B.OUTPUT_LABEL, B.OUTPUT_DATA]

        print("step | data label | output label | h")

        data_label_cnt = [0] * 10
        predict_label_cnt = [0] * 10

        for step in range(step_size):
            data = evaluation_batch.next_batch(mini_batch_size, key_list)
            feed_dict = {model["X"]: data[B.INPUT_DATA],
                         model["Y"]: data[B.OUTPUT_DATA],
                         model["Y_label"]: data[B.OUTPUT_LABEL],
                         model["conv_dropout_rate"]: 1,
                         model["fc_dropout_rate"]: 1,
                         model["is_training"]: False,
                         }

            label = data[B.OUTPUT_LABEL][0]
            predicted_label, h_ = sess.run([model[Model.PREDICTED_LABEL],
                                            model[Model.HYPOTHESIS]],
                                           feed_dict=feed_dict)

            predicted_label, h_ = predicted_label[0], h_[0]

            data_label_cnt[label] += 1
            if label == predicted_label:
                predict_label_cnt[label] += 1

            if step % 1000 == 0:
                print("process %d/%d" % (step, step_size))
                # print("%d   | %d | %d" % (step, label, predicted_label))
                # s = " ".join(["%.3f" % i for i in h_])
                # print(s)

        print(" label acc")
        for i in range(10):
            acc = predict_label_cnt[i] / data_label_cnt[i]
            print("%d %s    acc %f" % (i, label_name_list[i], acc))

        s = ["%d" % i for i in range(10)]
        s = " ".join(s)
        print(s)


        # this make clear all graphs
    tf.reset_default_graph()
    print("clean up")


if __name__ == '__main__':
    # for path in util.get_all_tuning_folder_path():
    #
    #     model_evaluation(path, Model_cnn_nn_softmax_B())

    print("test")
    evaluation_batch_config = Config(Config.OPTION_TEST_SET)
    evaluation_batch = Batch(evaluation_batch_config)
    print("evaluation batch loaded")

    path = ".\\save\\tuning\\2017-06-13_21_20_41.103668"
    # path = ".\\save\\model_A backUp\\2017-06-13_13_37_44.075433"

    model_evaluation(path, Model_cnn_nn_softmax_A(), evaluation_batch)

    print("train")
    evaluation_batch_config = Config(Config.OPTION_TRAIN_SET, 5)
    evaluation_batch = Batch(evaluation_batch_config)
    print("evaluation batch loaded")

    model_evaluation(path, Model_cnn_nn_softmax_A(), evaluation_batch)

    pass
