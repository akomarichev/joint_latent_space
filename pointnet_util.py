""" Joint Latent Space

Author: A. Komarichev
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/grouping_local'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/radius_estimation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import group_point, knn_point 
from tf_grouping_adaptive import query_ball_point_adaptive
from tf_radius_estimation import estimate_radiuses

import tensorflow as tf
import numpy as np
import tf_util
import pointfly as pf

SIGMA = 0.01
CLIP = 0.01

def sample_and_group(npoint, radius, nsample, xyz, normals, points, queries, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    indecies = farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, indecies)
    new_normals = gather_point(normals, indecies)

    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        # Estimate radiuses
        noise = tf.clip_by_value(SIGMA * tf.random.normal(tf.shape(xyz)), -1*CLIP, CLIP)
        xyz_noisy = xyz + noise
        new_xyz_noisy = gather_point(xyz_noisy, indecies)
        _, _, outi = pf.knn_indices_general(new_xyz_noisy, xyz_noisy, nsample, sort=True, unique=False)
        pts_cnt = tf.constant(np.tile(np.array([nsample]).reshape((1,1)), (32, npoint)), dtype=tf.int32)
        outr, centroids, dist, nu = estimate_radiuses(xyz_noisy, new_xyz_noisy, new_normals, outi, pts_cnt)
        outr = tf.clip_by_value(outr, 0, radius)
        idx, pts_cnt = query_ball_point_adaptive(outr, nsample, xyz, new_xyz)

    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization

    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_normals, new_points, idx, grouped_xyz

def adaptive_qb(xyz, normals, points, queries, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=False, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' Adaptive query ball module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            # nsample = xyz.get_shape()[1].value
            # new_xyz, new_normals, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
            if points is not None:
                new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
            else:
                new_points = xyz
            new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
            new_xyz = None
            new_normals = None
            idx = None
        else:
            new_xyz, new_normals, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, normals, points, queries, knn, use_xyz)

        # Point Feature Embedding
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True)
                new_points *= weights
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing 
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) 
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_normals, new_points, idx