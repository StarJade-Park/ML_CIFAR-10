from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps

import random
import numpy
import os
import time

def pickle(data, path):
    import pickle as pick
    with open(path, 'wb') as f:
        pick.dump(data, f)
    pass


def unpickle(path, encoding="ASCII"):
    import pickle as pick
    with open(path, 'rb') as f:
        data = pick.load(f, encoding=encoding)
    return data


def print_param(param):
    print("param info")
    print("#####################################")
    for param_name in param["param_list"]:
        if param_name is not "param_list":
            print("%s:" % param_name,
                  param[param_name])
    print("#####################################")
    print()


def pre_load():
    import tensorflow as tf
    a = tf.constant(1)
    with tf.Session() as sess:
        sess.run(a)
    tf.reset_default_graph()


def look_tunning():
    def filter(param):
        if param["learning_rate"] == 0.1 or param["learning_rate"] == 0.01:
            return False

        return True

    import os
    tuning_folder = os.path.join(".", "save", "tuning")

    for folder_name in os.listdir(tuning_folder):
        dir = os.path.join(tuning_folder, folder_name)
        check_point_dir = os.path.join(dir, "checkpoint")
        param_dir = os.path.join(dir, "param")
        out_dir = os.path.join(dir, "~epoch_000199")

        if os.path.exists(check_point_dir) is True:

            param = unpickle(param_dir)

            out = unpickle(out_dir)
            train_acc, test_acc, train_cost, test_cost = list(out)[-1]

            if not filter(param):
                continue

            print(dir)
            if train_acc > 0.40:
                print("### good")
                print(train_acc, test_acc, train_cost, test_cost)
                print_param(param)

            else:
                # print("### bad")
                # print(train_acc, test_acc, train_cost, test_cost)
                pass


        print()

    # saver_file_name = "check_point"
    # saver_path = os.path.join(folder_path, saver_file_name)
    #
    # param_file_name = "param"
    # param_file_path = os.path.join(folder_path, param_file_name)
    # util.pickle(param, param_file_path)
    # print("save param")
    #
    # model = cnn.build_model(param)
    # print("build model")
    #
    # epoch, out = train_and_test(model, param, saver_path)
    # path = os.path.join(folder_path, "~epoch_" + str(epoch).zfill(6))
    # util.pickle(out, path)
    pass


def raw2img(raw):
    img_mode = 'RGB'
    img_size = (32, 32)
    img = Image.new(img_mode, img_size)

    r, g, b = raw[:1024], raw[1024:1024 * 2], raw[1024 * 2:1024 * 3]
    rgb_list = list(zip(r, g, b))
    for x in range(32):
        for y in range(32):
            img.putpixel((x, y), rgb_list[y * 32 + x])

    return img


def img2raw(img):
    R, G, B = [], [], []
    for y in range(32):
        for x in range(32):
            r_pix, g_pix, b_pix = img.getpixel((x, y))
            # print(r_pix, g_pix, b_pix)
            R += [r_pix]
            G += [g_pix]
            B += [b_pix]

    return numpy.array(R + G + B)

def get_distored_data(data):
    img = raw2img(data)

    # flip horizonal
    if random.randint(0, 1) == 0:
        img = ImageOps.mirror(img)
    # crop

    # brightness
    bright_factor_min = 0.5
    bright_factor_max = 1.5
    bright_factor = random.uniform(bright_factor_min, bright_factor_max)
    img = ImageEnhance.Brightness(img).enhance(bright_factor)

    # contrast
    contrast_factor_min = 0.5
    contrast_factor_max = 1.5
    contrast_factor = random.uniform(contrast_factor_min, contrast_factor_max)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # save raw img
    return img2raw(img)

def gen_train_batch_distorted():
    BATCH_FILE_LABEL = b'batch_label'
    INPUT_DATA = b'data'
    INPUT_FILE_NAME = b'filenames'
    OUTPUT_LABEL = b'labels'
    OUTPUT_DATA = "output_list"

    DEFAULT_DIR_LIST = "dir_list"
    DEFAULT_NAME_KEY_LIST = "key_list"
    DEFAULT_TRAIN_DATA_BATCH_FILE_FORMAT = "data_batch_%d"
    DEFAULT_TRAIN_BATCH_FILE_NUMBER = 5
    DEFAULT_TEST_DATA_BATCH_FILE_FORMAT = "test_batch"

    # TODO LOOK at me this way is better
    DEFAULT_BATCH_FOLDER_DIR = os.path.join(".", "cifar-10-batches-py")

    distorted_data_batch_folder = "distorted_data_batch"
    for batch_num in range(1, 1 + 1):
        dir = os.path.join(DEFAULT_BATCH_FOLDER_DIR,
                           DEFAULT_TRAIN_DATA_BATCH_FILE_FORMAT % batch_num)

        print(dir)
        batch = unpickle(dir, encoding="bytes")
        size = len(batch[INPUT_DATA])
        for i in range(size):
            if i % 100 == 0:
                print("batch = %d file = %d" % (batch_num, i))

            # load raw img
            origin = batch[INPUT_DATA][i]
            distored = get_distored_data(origin)

        # new_dir = os.path.join(DEFAULT_BATCH_FOLDER_DIR,
        #                        "distored_data_batch_%d" % batch_num)
        # pickle(batch, new_dir)


if __name__ == '__main__':
    start = time.time()
    gen_train_batch_distorted()
    print(time.time()-start)
    # look_tunning()
