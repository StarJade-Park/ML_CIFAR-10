# ML_CIFAR-10 flags
import os
import tensorflow as tf
flags = tf.app.flags
Flags = flags.FLAGS

# TODO refactoring use tf.app.flags(mean: definition all flags)
# batch flags
flags.DEFINE_integer("image_size", 32 * 32 * 3, "image size of batch files")
flags.DEFINE_integer("label_number", 10, "label number of batch files")
# TODO batch resize
flags.DEFINE_integer("batch_size", 1000, "size of batch files")

# log flags
flags.DEFINE_integer("print_log_step_size", 1000, "print log step size")
flags.DEFINE_integer("summary_step_size", 10000, "summary step size")
flags.DEFINE_integer("checkpoint_step_size", 1000, "checkpoint step size")

# perceptron flags
flags.DEFINE_integer("perceptron_input_shape_size", 1000, "perceptron input shape size")
flags.DEFINE_integer("perceptron_output_shape_size", 1000, "perceptron output shape size")

# train & test flags
flags.DEFINE_float("learning_rate", 0.01, "learning rate")
flags.DEFINE_integer("max_train_step", 1, "max train step")
flags.DEFINE_integer("max_test_step", 1, "max test step")

# dir constant values
TEST = "test"
TRAIN = "train"
TENSOR_BOARD = "tensor_board"
CHECKPOINT = "checkpoint"
SAVE = "save"

# dir flags
# linux -> SAVE_DIR = './save/dir/type'
# windows -> SAVE_DIR = '.\\save\\dir\\type'

# Checkpoint dirs
flags.DEFINE_string("dir_train_checkpoint", os.path.join(".", SAVE, CHECKPOINT, TRAIN), "train checkpoint dir")
flags.DEFINE_string("dir_test_checkpoint", os.path.join(".", SAVE, CHECKPOINT, TEST), "test checkpoint dir")
# tensorboard dirs
flags.DEFINE_string("dir_train_tensorboard", os.path.join(".", SAVE, TENSOR_BOARD, TRAIN), "train tensorboard dir")
flags.DEFINE_string("dir_test_tensorboard", os.path.join(".", SAVE, TENSOR_BOARD, TEST), "test tensorboard dir")