""" Joint Latent Space

Author: A. Komarichev
"""

import os
import sys
import tensorflow as tf
import tf_util
import layers
from tflearn.layers.core import dropout
import numpy as np

# The number of samples per batch.
BATCH_SIZE = 32

# The height of each image.
IMG_HEIGHT = 128

# The width of each image.
IMG_WIDTH = 128

# The number of color channels per image.
IMG_CHANNELS = 3

# Feature vector dimension
IMG_FEATURES = 1024

ngf = 64


def get_outputs(inputs):
    input_img = inputs['input_img']
    input_features_img = inputs['input_features_img']
    is_training = inputs['is_training']

    with tf.variable_scope("AE_IMG") as scope:
        out_features_img = encoder_img(input_img, is_training, name="enc_img")
        out_img = decoder_img(out_features_img, is_training, name="dec_img")

        scope.reuse_variables()
        out_img_from_features = decoder_img(input_features_img, is_training, name="dec_img")

    return {
        'out_features_img': out_features_img,
        'out_img': out_img,
        'out_img_from_features': out_img_from_features
    }

def encoder_img(input_img, is_training, name="encoder_img"):
    with tf.variable_scope(name):
        batch_size = tf.shape(input_img)[0]

        #128 128
        net = tf_util.conv2d(input_img, ngf, [5,5], padding='SAME', is_training=is_training, scope='conv_enc_img_1', bn=True)
        net = tf_util.conv2d(net, ngf, [3,3], padding='SAME', stride=[2,2], is_training=is_training, scope='conv_enc_img_2', bn=True)
        #64 64
        net = tf_util.conv2d(net, 2*ngf, [3,3], padding='SAME', stride=[2,2], is_training=is_training, scope='conv_enc_img_3', bn=True)
        #32 32
        net = tf_util.conv2d(net, 4*ngf, [3,3], padding='SAME', stride=[2,2], is_training=is_training, scope='conv_enc_img_5', bn=True)
        #16 16
        net = tf_util.conv2d(net, 8*ngf, [3,3], padding='SAME', stride=[2,2], is_training=is_training, scope='conv_enc_img_7', bn=True)
        #8 8
        net = tf_util.conv2d(net, 16*ngf, [3,3], padding='SAME', stride=[2,2], is_training=is_training, scope='conv_enc_img_9', bn=True)

        # Symmetric function: max pooling
        net = tf.reduce_max(net, axis=[1,2], keepdims=True, name='maxpool_enc_img')
        net = tf.reshape(net, [batch_size, 1024])

        net = tf_util.fully_connected(net, 1024, IMG_FEATURES, bn=True, scope='fc1_enc_img', is_training=is_training)

        return net
        
def decoder_img(input_img_features, is_training, name="decoder_img"):
    with tf.variable_scope(name):
        batch_size = tf.shape(input_img_features)[0]

        net = tf_util.fully_connected(input_img_features, IMG_FEATURES, 4*4*1024, bn=True, scope='fc_gen_0', is_training=is_training)

        # DECONV Decoder
        net = tf.reshape(net, [batch_size, 4, 4, 1024])
        # 4x4x1024
        net = tf_util.conv2d_transpose(net, 16*ngf, kernel_size=[3,3], padding='SAME', stride=[2,2], is_training=is_training, scope='upconv_dec_img_1', bn=True)
        # 8x8x1024
        net = tf_util.conv2d_transpose(net, 8*ngf, kernel_size=[3,3], padding='SAME', stride=[2,2], is_training=is_training, scope='upconv_dec_img_2', bn=True)
        # 16x16x512
        net = tf_util.conv2d_transpose(net, 4*ngf, kernel_size=[3,3], padding='SAME', stride=[2,2], is_training=is_training, scope='upconv_dec_img_3', bn=True)
        # 32x32x256
        net = tf_util.conv2d_transpose(net, 2*ngf, kernel_size=[3,3], padding='SAME', stride=[2,2], is_training=is_training, scope='upconv_dec_img_4', bn=True)
        # 64x64x128
        net = tf_util.conv2d_transpose(net, ngf, kernel_size=[3,3], padding='SAME', stride=[2,2], is_training=is_training, scope='upconv_dec_img_5', bn=True)
        # 128x128x64
        net = tf_util.conv2d(net, 3, [5,5], padding='SAME', activation_fn=tf.nn.tanh, scope='conv_dec_img6')
        # 128x128x3

        return net