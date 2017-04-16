from util import *
from model import cGAN
from matplotlib.pyplot import *
from numpy import *
import os

flags = tf.app.flags
flags.DEFINE_integer('num_epochs', 20, 'Number of epochs')
flags.DEFINE_integer('batch_size', 2, 'Number of images used for each iteration')
flags.DEFINE_integer('year', 2016, 'Dataset for which year')
flags.DEFINE_bool('use_reverse', False, 'Whether to use the reversed sequences during training')
flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate for the Adam optimizer')
flags.DEFINE_float('xentropy_lambda', 100, 'Weight for the cross entropy loss')
flags.DEFINE_string('sample_dir', './samples', 'Where to store the results')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Where to store the checkpoints')
flags.DEFINE_string('log_dir', './logs', 'Where to store the logs')
flags.DEFINE_string('data_dir', '/mnt/data/DAVIS', 'Path to the DAVIS datasets')
flags.DEFINE_string('phase', 'train', 'Should be `train` or `val`')
flags.DEFINE_string('name', 'DAVIS', 'Name for this run')
flags.DEFINE_string('gpu', '0', 'GPU to be used')

config = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

def main(_):
    with tf.Session() as sess:
        mdl = cGAN(sess, config)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if mdl.phase is 'train':
            mdl.train()
        else:
            mdl.eval()

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
