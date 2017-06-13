import pickle
import numpy as np
import os
import sys
import random
import util
import threading
from multiprocessing import Process, Pipe

import queue
import time

BATCH_INIT_ERROR_MSG = "Config init error : option is None," \
                       " option must Config.OPOPTION_TRAIN_SET " \
                       "or Config.OPTION_TEST_SET"

DEFAULT_DIR_LIST = "dir_list"
DEFAULT_NAME_KEY_LIST = "key_list"
DEFAULT_TRAIN_DATA_BATCH_FILE_FORMAT = "data_batch_%d"
DEFAULT_TRAIN_BATCH_FILE_NUMBER = 2
DEFAULT_TEST_DATA_BATCH_FILE_FORMAT = "test_batch"
# DEFAULT_BATCH_FOLDER = ".\\cifar-10-batches-py" # windows
# DEFAULT_BATCH_FOLDER = "./cifar-10-batches-py" # ubuntu

# TODO LOOK at me this way is better
DEFAULT_BATCH_FOLDER_DIR = os.path.join(".", "cifar-10-batches-py")

# key string of batch data dict
BATCH_FILE_LABEL = b'batch_label'
INPUT_FILE_NAME = b'filenames'
INPUT_DATA = b'data'
OUTPUT_LABEL = b'labels'
OUTPUT_DATA = "output_list"


# TODO need refactoring
class Config:
    KEY_LIST = DEFAULT_NAME_KEY_LIST
    DIR_LIST = DEFAULT_DIR_LIST
    FOLDER_NAME = DEFAULT_BATCH_FOLDER_DIR
    TRAIN_BATCH_FILE_NAME_FORMAT = DEFAULT_TRAIN_DATA_BATCH_FILE_FORMAT
    TRAIN_BATCH_FILE_NUMBER = DEFAULT_TRAIN_BATCH_FILE_NUMBER
    TEST_BATCH_FILE_NAME_FORMAT = DEFAULT_TEST_DATA_BATCH_FILE_FORMAT

    OPTION_TRAIN_SET = "train_set"
    OPTION_TEST_SET = "test_set"

    def __init__(self, option=None):
        # init config
        self.config = dict()

        # set self.config[DEFAULT_DIR_LIST]
        self.option = option
        if option == self.OPTION_TRAIN_SET:
            # init dir_list
            self.config[DEFAULT_DIR_LIST] = []
            for i in range(1, self.TRAIN_BATCH_FILE_NUMBER + 1):
                self.config[DEFAULT_DIR_LIST] \
                    += [os.path.join(self.FOLDER_NAME, self.TRAIN_BATCH_FILE_NAME_FORMAT % i)]

        elif option == self.OPTION_TEST_SET:
            self.config[DEFAULT_DIR_LIST] \
                = [os.path.join(self.FOLDER_NAME, self.TEST_BATCH_FILE_NAME_FORMAT)]
        else:
            print(BATCH_INIT_ERROR_MSG)
            raise RuntimeError

        # set self.config[KEY_LIST]
        self.config[self.KEY_LIST] = [
            BATCH_FILE_LABEL,
            INPUT_DATA,
            INPUT_FILE_NAME,
            OUTPUT_LABEL,
            OUTPUT_DATA,
        ]


class Batch:
    def __init__(self, config):
        self.config = config.config
        self.option = config.option

        self.batch = {}
        self.load_batch()
        self.batch_index = 0
        self.batch_size = len(self.batch[INPUT_DATA])
        self.__generate_y_data()

        self.Q_PUSH_SIZE = 100
        self.MAX_Q_SIZE = 1024 * 16
        self.Q_SIZE = 1024 * 1
        self.Thread_exit = False
        self.Thread_run = False
        self.dict_q = {}
        self.init_thread()

        pass

    def init_thread(self):
        for key in self.batch:
            self.dict_q[key] = queue.Queue()

        thread = threading.Thread(target=self.thread_provide_distorted_data)
        thread.start()

    def get_q_size(self):
        return self.dict_q[INPUT_DATA].qsize()

    def thread_provide_distorted_data(self):
        key_list = [INPUT_DATA, OUTPUT_LABEL, OUTPUT_DATA]
        push_size = self.Q_PUSH_SIZE

        index = 0
        while True:

            if self.Thread_exit:
                break

            if self.dict_q[INPUT_DATA].qsize() > self.Q_SIZE:
                time.sleep(1)
                continue

            # push data
            if self.option == Config.OPTION_TRAIN_SET:
                # index = random.randint(0, self.batch_size - push_size - 1)
                index = (index + push_size) % self.batch_size
            else:
                index = (index + push_size) % self.batch_size

                # print(range(index, index + size))

            for i in range(index, index + push_size):
                for key in key_list:
                    if key is INPUT_DATA:
                        if self.option == Config.OPTION_TRAIN_SET:
                            self.dict_q[key].put_nowait(util.get_distorted_data(self.batch[key][i]))
                        else:
                            self.dict_q[key].put_nowait(util.get_cropped_data(self.batch[key][i]))
                    else:
                        self.dict_q[key].put_nowait(self.batch[key][i])

    def stop_thread(self):
        self.Thread_run = False

    def resume_thread(self):
        self.Thread_run = True

    def thread_exit(self):
        self.Thread_exit = True

    def __append(self, a, b, key):
        if key == BATCH_FILE_LABEL:
            return a + self.__assign(b, key)
        elif key == INPUT_DATA:
            return np.concatenate((a, b))
        elif key == INPUT_FILE_NAME:
            return a + b
        elif key == OUTPUT_LABEL:
            return a + b

    @staticmethod
    def __assign(a, key):
        if key == BATCH_FILE_LABEL:
            return [a]
        elif key == INPUT_DATA:
            return a
        elif key == INPUT_FILE_NAME:
            return a
        elif key == OUTPUT_LABEL:
            return a

    # load_batch(self)
    # load batch file every file_dir in self.config["dir_list"]
    def load_batch(self):
        for dir_ in self.config[DEFAULT_DIR_LIST]:
            data = util.unpickle(dir_, encoding='bytes')
            for key in data:
                if key in self.batch:
                    self.batch[key] = self.__append(self.batch[key], data[key], key)
                else:
                    self.batch[key] = self.__assign(data[key], key)

    # __generate_y_data(self)
    # generate y_label from y_data
    # ex) y_label = 3, result
    # y_data = [0,0,0,1,0,0,0,0,0,0]
    def __generate_y_data(self):
        self.batch[OUTPUT_DATA] = []
        for idx in self.batch[OUTPUT_LABEL]:
            temp = [0 for _ in range(10)]
            temp[idx] = 1
            self.batch[OUTPUT_DATA] += [temp]
        pass

    # reset_batch_index(self)
    # reset batch index
    # same as self.batch_index = 0
    def reset_batch_index(self):
        self.batch_index = 0

    # next_batch(self, size, key_list=None)
    # iter next batch dict from self.batch_index to self.batch_index + size
    # size : size of result iterated batch
    # key_list : select key of batch list
    # default is
    # ex) key_list = ["X", "Y"] result is
    # {"X" = [...], "Y" = [...]}
    def next_distorted_batch(self, size, key_list=None):
        if key_list is None:
            key_list = self.config[Config.KEY_LIST]

        batch = {}
        for key in key_list:
            part = []
            cnt = size
            while cnt > 0:
                if self.dict_q[key].qsize() > 0:
                    part += [self.dict_q[key].get_nowait()]
                    cnt -= 1
                else:
                    time.sleep(0.01)

            self.dict_q[key].task_done()
            batch[key] = part

        return batch

    def next_batch(self, size, key_list=None):
        if key_list is None:
            key_list = self.config[Config.KEY_LIST]

        index = self.batch_index
        self.batch_index = (self.batch_index + size) % self.batch_size

        batch = {}
        for key in key_list:
            part = self.batch[key][index:index + size]
            batch[key] = part[:size]

        return batch


# # TODO add decorator
# def ex_batch_train_set():
#     print("ex_batch_train_set ")
#     print("*****************************************")
#     batch_config = Config(option=Config.OPTION_TRAIN_SET)
#     # print(batch_config.config["dir_list"])
#
#     b = Batch(batch_config)
#
#     size = 3
#     for _ in range(1):
#         key_list = [b'data', b'labels', "output_list"]
#         batch = b.next_batch(size, key_list)
#         for key in batch:
#             print(key)
#             for i in batch[key]:
#                 print(i)
#                 print()
#
#     print("*****************************************")
#     return
#
#
# # TODO add decorator
# def ex_batch_test_set():
#     print()
#     print("*****************************************")
#     batch_config = Config(option=Config.OPTION_TEST_SET)
#     # print(batch_config.config["dir_list"])
#
#     b = Batch(batch_config)
#
#     size = 3
#     for _ in range(1):
#         key_list = [b'data', b'labels', "output_list"]
#         batch = b.next_batch(size, key_list)
#         for key in batch:
#             print(key)
#             for i in batch[key]:
#                 print(i)
#                 print()
#     print("*****************************************")
#     return
#

if __name__ == '__main__':
    # ex_batch_train_set()
    # ex_batch_test_set()
    import time

    config = Config(Config.OPTION_TRAIN_SET)
    batch = Batch(config)
    print("batch loaded")
    batch.resume_thread()

    key_list = [INPUT_DATA, OUTPUT_LABEL, OUTPUT_DATA]
    for i in range(1000):
        start = time.time()
        q = batch.next_batch_from_q(8, key_list)
        print(time.time() - start)
    time_stamp = time.localtime(time.time() - start)

    # print(time.strftime("%H:%M:%S", time_stamp))
    print(q)
    pass
