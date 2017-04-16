import tensorflow as tf
from numpy import load

VGG_MEAN = [103.939, 116.779, 123.68]
vgg_weights = load('./vgg16.npy', encoding='latin1').item()

def max_pool(x, name):
    return tf.layers.max_pooling2d(x, 2, 2, padding='same', name=name)


def conv2d_relu(x, reuse=None, name='conv2d'):
    kernel = vgg_weights[name][0]
    bias = vgg_weights[name][1]

    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, kernel.shape[-1], kernel.shape[:2],
                        padding='same', reuse=reuse, name='conv2d',
                        kernel_initializer=tf.constant_initializer(kernel),
                        bias_initializer=tf.constant_initializer(bias))

        return tf.nn.relu(x, name='relu')


def VGG16(rgb, reuse=None):
    # Convert RGB to BGR
    red, green, blue = tf.unstack(rgb * 255, 3, axis=-1)
    bgr = tf.stack([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], axis=-1)

    conv1_1 = conv2d_relu(bgr, reuse=reuse, name='conv1_1')
    conv1_2 = conv2d_relu(conv1_1, reuse=reuse, name='conv1_2')
    pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 = conv2d_relu(pool1, reuse=reuse, name='conv2_1')
    conv2_2 = conv2d_relu(conv2_1, reuse=reuse, name='conv2_2')
    pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 = conv2d_relu(pool2, reuse=reuse, name='conv3_1')
    conv3_2 = conv2d_relu(conv3_1, reuse=reuse, name='conv3_2')
    conv3_3 = conv2d_relu(conv3_2, reuse=reuse, name='conv3_3')
    pool3 = max_pool(conv3_3, 'pool3')

    conv4_1 = conv2d_relu(pool3, reuse=reuse, name='conv4_1')
    conv4_2 = conv2d_relu(conv4_1, reuse=reuse, name='conv4_2')
    conv4_3 = conv2d_relu(conv4_2, reuse=reuse, name='conv4_3')
    pool4 = max_pool(conv4_3, 'pool4')

    conv5_1 = conv2d_relu(pool4, reuse=reuse, name='conv5_1')
    conv5_2 = conv2d_relu(conv5_1, reuse=reuse, name='conv5_2')
    conv5_3 = conv2d_relu(conv5_2, reuse=reuse, name='conv5_3')
    pool5 = max_pool(conv5_3, 'pool5')

    return pool1, pool2, pool3, pool4, pool5
