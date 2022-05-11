''' Radius Estimation
Author: Artem Komarichev
All Rights Reserved. 2022
'''
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
radius_estimation_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_radius_estimation_so.so'))

sys.path.append(os.path.join(ROOT_DIR, '../utils'))

def estimate_radiuses(input_xyz, query_xyz, query_normals, idx, pts_cnt):
    '''
    Input:
        input_xyz: (batch_size, ndataset, c) float32 array, input points
        query_xyz: (batch_size, npoint, c) float32 array, query points
        query_normals: (batch_size, npoint, c) float32 array, query normals
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
        idx: (batch_size, npoint, k) int32 array, indecies of the k closest points
    Output:
        outr: (batch_size, npoint) float32 array, estimated radiuses based on the local geometry.
    '''
    return radius_estimation_module.radius_estimation_org(input_xyz, query_xyz, query_normals, idx, pts_cnt)
ops.NoGradient("RadiusEstimationOrg")

def clip_radiuses(radiuses, clipped_values):
    return radius_estimation_module.clip_radius_org(radiuses, clipped_values)
ops.NoGradient("ClipRadiusOrg")

if __name__=='__main__':
    pass