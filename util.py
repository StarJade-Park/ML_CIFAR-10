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
    for param_name in param:
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
