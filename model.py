import os
from util import *
from numpy import *
from glob import glob
from vgg16 import VGG16
from scipy.misc import imread, imsave
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage.filters import median_filter

class cGAN(object):
    def __init__(self, sess, config):
        self.sess = sess

        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.learning_rate = config.learning_rate
        self.xentropy_lambda = config.xentropy_lambda
        self.phase = config.phase
        self.year = config.year
        self.use_reverse = config.use_reverse

        if config.phase == 'train':
            self.width = 832
            self.height = 480
        else:
            self.width = None
            self.height = None

        self.name = config.name
        self.ckpt_dir = config.checkpoint_dir + '/' + config.name
        self.log_dir = config.log_dir + '/' + config.name
        for d in [self.ckpt_dir, self.log_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
        self.sample_dir = '%s/%s/%d' % (config.sample_dir, config.name, config.year)
        self.data_dir = config.data_dir

        self.build_model()

        self.saver = tf.train.Saver(max_to_keep=25)
        self.sess.run(tf.global_variables_initializer())
        self.load()

    def input_pipeline(self, fn_seg1, fn_img0, fn_img1):
        reader = tf.WholeFileReader()

        with tf.variable_scope('segmentation'):
            # Frame 1
            queue_seg1 = tf.train.string_input_producer(fn_seg1, shuffle=False)
            _, value = reader.read(queue_seg1)
            seg1 = tf.image.decode_png(value, channels=1)
            seg1 = tf.cast(tf.not_equal(seg1, 0), tf.float32)

        with tf.variable_scope('image'):
            # Frame 0
            queue_img0 = tf.train.string_input_producer(fn_img0, shuffle=False)
            _, value = reader.read(queue_img0)
            img0 = tf.image.decode_jpeg(value, channels=3)
            img0 = tf.cast(img0, tf.float32) / 255.0

            # Frame 1
            queue_img1 = tf.train.string_input_producer(fn_img1, shuffle=False)
            _, value = reader.read(queue_img1)
            img1 = tf.image.decode_jpeg(value, channels=3)
            img1 = tf.cast(img1, tf.float32) / 255.0

        data = tf.concat([img0, img1, seg1], axis=-1)

        data = tf.image.resize_image_with_crop_or_pad(data, self.height, self.width)

        return data

    def train(self):
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        sum_merged = tf.summary.merge_all()

        total_count = 0
        print('Start training with %d images...' % self.num_images)
        t_start = time.time()
        for epoch in range(self.num_epochs):
            for step in range(1 + self.num_images // self.batch_size):
                # Update D network
                self.sess.run(self.optim_D)

                # Update G network
                self.sess.run(self.optim_G)

                if total_count % (self.num_images // self.batch_size // 20) == 0:
                    loss_D, loss_G, summary = self.sess.run([self.loss_D, self.loss_G, sum_merged])
                    writer.add_summary(summary, total_count)

                    m, s = divmod(time.time() - t_start, 60)
                    h, m = divmod(m, 60)
                    print('Epoch: [%4d/%4d] [%4d/%4d], time: [%02d:%02d:%02d], loss_D: %.4f, loss_G: %.4f'
                        % ( epoch, self.num_epochs, step, 1 + self.num_images // self.batch_size, h, m, s, loss_D, loss_G))
                total_count += 1

            self.saver.save(self.sess, os.path.join(self.ckpt_dir, 'model.ckpt'), global_step=epoch + 1)

    def build_model(self):
        self.fn_seg0 = []
        self.fn_seg1 = []
        self.fn_img0 = []
        self.fn_img1 = []
        with open('%s/ImageSets/%d/%s.txt' % (self.data_dir, self.year, self.phase), 'r') as f:
            for line in f:
                self.fn_img0 += sorted(glob('%s/JPEGImages/480p/%s/*.jpg' % (self.data_dir, line[:-1])))[:-1]
                self.fn_img1 += sorted(glob('%s/JPEGImages/480p/%s/*.jpg' % (self.data_dir, line[:-1])))[1:]
                self.fn_seg0 += sorted(glob('%s/Annotations/480p/%s/*.png' % (self.data_dir, line[:-1])))[:-1]
                self.fn_seg1 += sorted(glob('%s/Annotations/480p/%s/*.png' % (self.data_dir, line[:-1])))[1:]

        self.num_images = len(self.fn_seg1)

        with tf.variable_scope('data_loader'):
            if self.phase == 'train':
                if self.use_reverse:
                    print('Using both forward and reversed sequences.')
                    data = self.input_pipeline(self.fn_seg1 + self.fn_seg0,
                                                self.fn_img0 + self.fn_img1,
                                                self.fn_img1 + self.fn_img0)
                    self.num_images *= 2
                else:
                    print('Using only forward sequences.')
                    data = self.input_pipeline(self.fn_seg1, self.fn_img0, self.fn_img1)

                data = tf.image.random_flip_left_right(data)
                min_queue_examples = 512
                data = tf.train.shuffle_batch([data],
                            batch_size=self.batch_size,
                            num_threads=8,
                            capacity=min_queue_examples + 3 * self.batch_size,
                            min_after_dequeue=min_queue_examples,
                            allow_smaller_final_batch=True)
                self.x = tf.slice(data, [0, 0, 0, 0], [-1, -1, -1, 6], name='img01')
                self.y = tf.slice(data, [0, 0, 0, 6], [-1, -1, -1, 1], name='seg1')
            elif self.phase == 'val':
                self.x = tf.placeholder(tf.float32, [1, self.height, self.width, 6])
                self.y = tf.placeholder(tf.float32, [1, self.height, self.width, 1])

        G, logits_G = self.FCN(self.x)
        self.sample, _ = self.FCN(self.x, training=False, reuse=True)
        D_real, logits_D_real = self.discriminator(self.x, self.y)
        D_fake, logits_D_fake = self.discriminator(self.x, G, reuse=True)

        with tf.variable_scope('loss_xentropy'):
            self.loss_xentropy = cross_entropy(logits_G, self.y)

        with tf.variable_scope('loss_G'):
            self.loss_G = cross_entropy(logits_D_fake, tf.ones_like(D_fake))

        with tf.variable_scope('loss_D'):
            loss_D_real = cross_entropy(logits_D_real, tf.ones_like(D_real))
            loss_D_fake = cross_entropy(logits_D_fake, tf.zeros_like(D_fake))
            self.loss_D = loss_D_real + loss_D_fake

        y3 = tf.tile(self.y, [1, 1, 1, 3])
        s3 = tf.tile(self.sample, [1, 1, 1, 3])
        self.results = tf.concat([self.x[..., 3:], y3, s3], axis=1)
        self.results = tf.cast(tf.round(self.results * 255), tf.uint8)

        # summaries
        tf.summary.image('results', self.results)
        tf.summary.histogram('D/real', D_real)
        tf.summary.histogram('D/fake', D_fake)

        tf.summary.scalar('loss/xentropy', self.loss_xentropy)
        tf.summary.scalar('loss/G', self.loss_G)
        tf.summary.scalar('loss/D', self.loss_D)

        t_vars = tf.trainable_variables()
        for v in t_vars:
            tf.summary.histogram(v.name, v)

        vars_G = [var for var in t_vars if 'FCN' in var.name]
        vars_D = [var for var in t_vars if 'discriminator' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope('optimizers'):
                self.optim_D = tf.train.AdamOptimizer(self.learning_rate,
                                    beta1=0.5).minimize(self.loss_D, var_list=vars_D)
                self.optim_G = tf.train.AdamOptimizer(self.learning_rate,
                                    beta1=0.5).minimize(self.loss_G + self.xentropy_lambda * self.loss_xentropy, var_list=vars_G)

    def FCN(self, x, training=True, reuse=None):
        with tf.variable_scope('FCN') as scope:
            with tf.variable_scope('VGG16'):
                pool_f0 = VGG16(x[..., :3], reuse=reuse)
                pool_f1 = VGG16(x[..., 3:], reuse=True)

            e = []
            for k in range(5):
                e.append(tf.concat([pool_f0[k], pool_f1[k]], axis=-1, name='e%d' % k))
                e[-1] = tf.layers.dropout(e[-1], training=training, name='dropout%d' % k)

            d1 = upconv_bn_relu(e[4], 512, reuse=reuse, training=training, name='d1')
            d1 = tf.concat([d1, e[3]], axis=-1, name='d1/concat')

            d2 = upconv_bn_relu(d1, 256, reuse=reuse, training=training, name='d2')
            d2 = tf.concat([d2, e[2]], axis=-1, name='d2/concat')

            d3 = upconv_bn_relu(d2, 128, reuse=reuse, training=training, name='d3')
            d3 = tf.concat([d3, e[1]], axis=-1, name='d3/concat')

            d4 = upconv_bn_relu(d3, 64, reuse=reuse, training=training, name='d4')
            d4 = tf.concat([d4, e[0]], axis=-1, name='d4/concat')

            d5 = tf.layers.conv2d_transpose(d4, 1, 4, 2, padding='same', reuse=reuse, name='d5')

            return tf.nn.sigmoid(d5), d5


    def discriminator(self, x, y, reuse=None):
        with tf.variable_scope('discriminator') as scope:
            num_filters = 64
            h0 = conv_bn(tf.concat([x, y], axis=-1, name='h0/xy'), num_filters, reuse=reuse, name='h0')
            h1 = conv_bn(leakyReLU(h0, name='h1/lrelu'), 2 * num_filters, reuse=reuse, name='h1')
            h2 = conv_bn(leakyReLU(h1, name='h2/lrelu'), 4 * num_filters, reuse=reuse, name='h2')
            h3 = conv_bn(leakyReLU(h2, name='h3/lrelu'), 8 * num_filters, reuse=reuse, name='h3')
            h4 = conv_bn(leakyReLU(h3, name='h4/lrelu'), 8 * num_filters, reuse=reuse, name='h4')
            h5 = tf.layers.conv2d(leakyReLU(h4, name='h5/lrelu'), 1, 4, 2, reuse=reuse, padding='same', name='h5/conv2d')

            return tf.nn.sigmoid(h5), h5

    def load(self):
        print(' [*] Reading checkpoints from `%s` ...' % self.ckpt_dir)

        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,
                            os.path.join(self.ckpt_dir, ckpt_name))
            print(' [*] Success to read %s' % ckpt_name)
        else:
            if self.phase == 'train':
                print(' [*] Failed to find a checkpoint')
            else:
                raise ValueError(' [*] Failed to find a checkpoint')

    def eval(self):
        for k in range(self.num_images):
            img0 = imread(self.fn_img0[k])
            img1 = imread(self.fn_img1[k])

            h, w = img0.shape[:2]
            x = zeros([1, int(32 * ceil(h / 32)), int(32 * ceil(w / 32)), 6])
            x[:, :h, :w, :3] = img0 / 255
            x[:, :h, :w, 3:] = img1 / 255
            pred = self.sess.run(self.sample, feed_dict={self.x: x})[0, :h, :w, 0] > 0.5

            #print(self.fn_img0[k])
            #print(self.fn_img1[k])

            fn_split = self.fn_seg1[k].split('/')
            print('[%4d/%4d] %s/%s' % (k + 1, self.num_images, fn_split[-2], fn_split[-1]))

            folder = self.sample_dir + '/' + fn_split[-2]
            if not os.path.exists(folder):
                os.makedirs(folder)

            #pred = binary_dilation(pred, iterations=1)
            #pred = binary_erosion(pred, iterations=2)
            #pred = binary_dilation(pred, iterations=2)
            #pred = binary_erosion(pred, iterations=1)

            pred = median_filter(pred, 9)

            imsave('%s/%s' % (folder, fn_split[-1]), pred)
