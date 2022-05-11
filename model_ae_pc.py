"""Code for constructing the model and get the outputs from the model."""

import os
import sys
import tensorflow as tf
import tf_util
import layers
from tflearn.layers.core import dropout
import numpy as np
sys.path.append(os.path.dirname(os.getcwd()))
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point

# The number of samples per batch.
BATCH_SIZE = 32

# The number of points in an point cloud
PC_POINTS = 4096

# Number of centroids
# N_CENTROIDS = 1024

# The number of channels (e.g. xyz) per object
PC_CHANNELS = 3

# Feature vector dimension
PC_FEATURES = 1024


def get_outputs(inputs):
    input_pc = inputs['input_pc']
    input_features_pc = inputs['input_features_pc']
    is_training = inputs['is_training']

    with tf.variable_scope("AE_PC") as scope:
        out_features_pc = encoder_pc(input_pc, is_training, name="enc_pc")
        out_pc = decoder_pc(out_features_pc, is_training, name="dec_pc")

        scope.reuse_variables()
        out_pc_from_features = decoder_pc(input_features_pc, is_training, name="dec_pc")

    return {
        'out_features_pc': out_features_pc,
        'out_pc': out_pc,
        'out_pc_from_features': out_pc_from_features
    }

def encoder_pc(input_pc, is_training, name="encoder_pc"):
    with tf.variable_scope(name):
        batch_size = tf.shape(input_pc)[0]
        input_image = tf.expand_dims(input_pc, -1)

        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3], padding='VALID', stride=[1,1], bn=True, scope='conv_enc_pc_1', is_training=is_training)
        net = tf_util.conv2d(net, 128, [1,1], padding='SAME', stride=[1,1], bn=True, scope='conv_enc_pc_2', is_training=is_training)
        net = tf_util.conv2d(net, 256, [1,1], padding='SAME', stride=[1,1], bn=True, scope='conv_enc_pc_3', is_training=is_training)
        net = tf_util.conv2d(net, 512, [1,1], padding='SAME', stride=[1,1], bn=True, scope='conv_enc_pc_4', is_training=is_training)
        net = tf_util.conv2d(net, 1024, [1,1], padding='SAME', stride=[1,1], bn=True, scope='conv_enc_pc_5', is_training=is_training)

        # Symmetric function: max pooling
        net = tf.reduce_max(net, axis=[1,2], keepdims=True, name='maxpool_enc_pc')
        net = tf.reshape(net, [batch_size, 1024])

        net = tf_util.fully_connected(net, 1024, 1024, bn=True, activation_fn=tf.nn.tanh, scope='fc1_enc', is_training=is_training)

        return net
        
def decoder_pc(input_pc_features, is_training, name="decoder_pc"):
    with tf.variable_scope(name):
        batch_size = tf.shape(input_pc_features)[0]

        net = tf_util.fully_connected(input_pc_features, 1024, 1024, bn=True, scope='fc1_dec', is_training=is_training)
        net = tf_util.fully_connected(net, 1024, 1024, bn=True, scope='fc2_dec', is_training=is_training)
        net = tf_util.fully_connected(net, 1024, PC_POINTS * 3, activation_fn=tf.nn.tanh, scope='fc3')
        output = tf.reshape(net, (batch_size, PC_POINTS, 3))

        return output