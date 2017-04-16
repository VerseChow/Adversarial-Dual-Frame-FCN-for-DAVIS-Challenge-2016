from numpy import *
import tensorflow as tf
from matplotlib.pyplot import *
import time

def conv_bn(x, num_filters, training=True, reuse=None, name='conv_bn'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, num_filters, 4, 2,
                padding='same', use_bias=False, reuse=reuse, name='conv2d')
        x = tf.layers.batch_normalization(x, training=training, reuse=reuse,
                scale=False, epsilon=1e-6, momentum=0.999, name='bn')
        return x

def upconv_bn_relu(x, num_filters, training=True, reuse=None, name='upconv'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, num_filters, 4, 2,
                padding='same', use_bias=False, reuse=reuse, name='conv2d_transpose')
        x = tf.layers.batch_normalization(x, training=training, reuse=reuse,
                scale=False, epsilon=1e-6, momentum=0.999, name='bn')
        return tf.nn.relu(x, name='relu')

def leakyReLU(x, alpha=0.2, name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(alpha * x, x)

def cross_entropy(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
