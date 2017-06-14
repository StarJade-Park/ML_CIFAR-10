from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps

import random
import numpy
import os
import Batch

from slacker import Slacker

import time
import datetime
import model


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


def get_all_tuning_folder_path():
    folder_path_list = []
    tuning_folder_dir = os.path.join(".", "save", "tuning")
    for folder_name in os.listdir(tuning_folder_dir):
        folder_path_list += [os.path.join(tuning_folder_dir, folder_name)]

    return folder_path_list


def get_last_global_epoch(folder_path):
    import tensorflow as tf
    from os.path import join

    cnn = model.Model_cnn_nn_softmax_A()
    cnn_model = cnn.build_model(load_param(folder_path))

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, join(folder_path, "check_point"))
        global_epoch = sess.run(cnn_model["global_epoch"])

    tf.reset_default_graph()

    return global_epoch - 1


def pre_load():
    import tensorflow as tf

    a = tf.constant(1)
    with tf.Session() as sess:
        sess.run(a)
    tf.reset_default_graph()
    print("tesorflow loaded")


def look_tuning():
    import os
    tuning_folder = os.path.join(".", "save", "tuning")

    for folder_name in os.listdir(tuning_folder):
        folder_path = os.path.join(tuning_folder, folder_name)
        train_acc, test_acc, train_cost, test_cost = load_last_output(folder_path)[-1]

        if train_acc > 0.45 and train_cost > 0.5:
            print_last_output(folder_path)

            print_param(load_param(folder_path))
            print()

    pass


def raw2img(raw, size, img_mode="RGB"):
    img_size = (size, size)
    img = Image.new(img_mode, img_size)

    raw_size = size * size
    r, g, b = raw[:raw_size], raw[raw_size:raw_size * 2], raw[raw_size * 2:raw_size * 3]
    rgb_list = list(zip(r, g, b))
    for x in range(size):
        for y in range(size):
            img.putpixel((x, y), rgb_list[y * size + x])

    return img


def img2raw(img, size):
    R, G, B = [], [], []

    for y in range(size):
        for x in range(size):
            r_pix, g_pix, b_pix = img.getpixel((x, y))
            # print(r_pix, g_pix, b_pix)
            R += [r_pix]
            G += [g_pix]
            B += [b_pix]

    return numpy.array(R + G + B)


def crop_img(img, crop_size, center=False):
    if center:
        x = 4
        y = 4
    else:
        x = random.randint(0, 7)
        y = random.randint(0, 7)

    return img.crop((x, y, x + crop_size, y + crop_size))


def get_cropped_data(data, size=32, crop_size=24):
    return img2raw(crop_img(raw2img(data, size), crop_size, center=True), crop_size)


def get_distorted_data(data):
    img = raw2img(data, 32)

    # flip horizonal
    if random.randint(0, 1) == 0:
        img = ImageOps.mirror(img)

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

    # resize
    img = img.resize((40, 40))

    angle = random.randint(-30, 30)
    img = img.rotate(angle)

    # crop
    img = crop_img(img, 32)

    raw = img2raw(img, 32)

    return raw


def save_param(param, folder_path):
    param_file_path = os.path.join(folder_path, "param")
    pickle(param, param_file_path)
    print("save param", param_file_path)


def load_param(folder_path):
    param = unpickle(os.path.join(folder_path, "param"))
    print("load param", folder_path)
    return param


def save_output(output, epoch, folder_path):
    output_path = os.path.join(folder_path, "~epoch_" + str(epoch).zfill(6))
    pickle(output, output_path)
    print("save output", output_path)


def load_last_output(folder_path):
    last_epoch = get_last_global_epoch(folder_path)
    out_dir = os.path.join(folder_path, "~epoch_" + str(last_epoch).zfill(6))
    return unpickle(out_dir)[-1]


def get_new_tuning_folder():
    folder_name = str(datetime.datetime.utcnow()).replace(" ", "_").replace(":", "_")
    print(folder_name)
    folder_path = os.path.join(".", "save", "tuning", folder_name)
    print(folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("make dir")
    return folder_path


def print_param(param):
    ret = ""
    ret += "param info\n"
    ret += "#####################################\n"
    for param_name in param["param_list"]:
        if param_name is not "param_list":
            ret += "%s: %s\n" % (param_name, param[param_name])
    ret += "#####################################\n\n"

    print(ret)
    return ret


def print_last_output(path):
    train_acc, test_acc, train_cost, test_cost = load_last_output(path)
    ret = ""
    ret += "last output info\n"
    ret += "train_acc, test_acc, train_cost, test_cost\n"
    ret += "%.4f      %.4f    %.4f       %.4f\n" % (train_acc, test_acc, train_cost, test_cost)

    print(ret)
    return ret


def print_last_global_epoch(path):
    ret = "last epoch:\n", get_last_global_epoch(path)
    print(ret)
    return ret


def slack_bot(massage):
    token = "xoxb-194637315491-YE9sLMtbmpFNP1HwerUIcmmi"
    slack = Slacker(token)
    slack.chat.post_message('#ml-com-bot', massage)
    return


def time_stamp():
    # TODO implement this
    return


def strformat():
    # TODO implement this

    return


def print_epoch_log(path):
    epoch_num = 0
    for dir_ in os.listdir(path):
        if "~epoch" in dir_:
            epoch_log_path = os.path.join(path, dir_)
            for a, b, c, d in unpickle(epoch_log_path):
                s = "%d %.3f %.3f %.3f %.3f" % (epoch_num, a, b, c, d)
                print(s)
                epoch_num += 1


if __name__ == '__main__':
    start = time.time()

    # look_tuning()

    for path in get_all_tuning_folder_path():
        print(path)
        print_epoch_log(path)
