import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
grouping_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_grouping_adaptive_so.so'))
def query_ball_point_adaptive(radius, nsample, xyz1, xyz2):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    return grouping_module.query_ball_point_adaptive(xyz1, xyz2, radius, nsample)
ops.NoGradient('QueryBallPointAdaptive')

if __name__=='__main__':
    knn=True
    import numpy as np
    import time
    np.random.seed(100)
    pts = np.random.random((32,512,64)).astype('float32')
    tmp1 = np.random.random((32,512,3)).astype('float32')
    tmp2 = np.random.random((32,128,3)).astype('float32')
    with tf.device('/gpu:1'):
        points = tf.constant(pts)
        xyz1 = tf.constant(tmp1)
        xyz2 = tf.constant(tmp2)
        radius = 0.1 
        nsample = 64
        if knn:
            _, idx = knn_point(nsample, xyz1, xyz2)
            grouped_points = group_point(points, idx)
        else:
            idx, _ = query_ball_point(radius, nsample, xyz1, xyz2)
            grouped_points = group_point(points, idx)
            #grouped_points_grad = tf.ones_like(grouped_points)
            #points_grad = tf.gradients(grouped_points, points, grouped_points_grad)
    with tf.Session('') as sess:
        now = time.time() 
        for _ in range(100):
            ret = sess.run(grouped_points)
        #print time.time() - now
        #print ret.shape, ret.dtype
        #print ret
    
    
