"""Contains losses used for performing image-to-image domain adaptation."""
import tensorflow as tf
import sys, os
import model_ae_pc_adaptive as model
sys.path.append(os.path.dirname(os.getcwd()))
from tf_ops.nn_distance import tf_nndistance
from tf_ops.approxmatch import tf_approxmatch
import pointfly as pf
from pointnet_util import gather_point
import numpy as np

LOCAL_DISK_RADIUS = 0.004

SIGMA = 0.002
CLIP = 0.002

def L1_loss(real_images, generated_images):
    return tf.reduce_mean(tf.abs(real_images - generated_images))

def chamfer(y_pred, y):
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(y, y_pred)
    dists_forward_r=tf.reduce_mean(dists_forward)
    dists_backward_r=tf.reduce_mean(dists_backward)
    cd_loss=(dists_forward_r+dists_backward_r)
    return cd_loss, dists_forward, dists_backward

def emd(y_pred, y):
    match = tf_approxmatch.approx_match(y_pred, y)
    emd_loss = tf.reduce_mean(tf_approxmatch.match_cost(y_pred, y, match))
    return emd_loss

def particle_loss(y_pred, y_gt, normals_gt, surface_area):
    batch_size = tf.shape(y_pred)[0]

    # Find for each predicted point corresponding gt point with its normals
    _, _, _, idx = tf_nndistance.nn_distance(y_gt, y_pred)
    normals = gather_point(normals_gt, idx)
    queries = gather_point(y_gt, idx)

    # Projection on gt
    edge = tf.subtract(y_pred, queries)
    v = tf.multiply(tf.tile(tf.reduce_sum(tf.multiply(normals,edge), axis=-1, keep_dims=True), [1,1,3]), normals)
    proj_x = y_pred - v

    nsample = 21
    @tf.custom_gradient
    def proj_gradients(pred):
        def grad(dy):
            return dy - tf.multiply(tf.tile(tf.reduce_sum(tf.multiply(normals, dy), axis=-1, keep_dims=True), [1,1,3]), normals)
        return tf.identity(pred), grad

    corrected_proj_x = proj_gradients(proj_x)
    sigma = tf.tile(tf.reshape(0.3 * tf.math.sqrt(surface_area/tf.to_float(model.PC_POINTS)), (batch_size, 1, 1)), [1,model.PC_POINTS, nsample-1])
    dist, _, idx = pf.knn_indices_general(corrected_proj_x, corrected_proj_x, nsample)
    particle_loss = tf.reduce_mean(tf.reduce_sum(tf.exp(-tf.math.divide(dist[:,:,1:], 4*(tf.square(sigma)))), axis=-1))

    return 5.0 * particle_loss, corrected_proj_x, proj_x, queries, normals

def evaluate_uformity(y_pred, y_gt):
    # Evaluate predicted
    _ , _, idx = pf.knn_indices_general(y_pred, y_pred, 2)
    diff_pred = gather_point(y_pred, idx[:,:,1]) - y_pred
    closest_dist_pred = tf.sqrt(tf.reduce_sum(tf.square(diff_pred), -1))
    mean_dist_pred = tf.reduce_mean(closest_dist_pred, axis=[1])
    std_dist_pred = tf.math.reduce_std(closest_dist_pred, axis=[1])

    # Evaluate gt
    _ , _, idx = pf.knn_indices_general(y_gt, y_gt, 2)
    diff_gt = gather_point(y_gt, idx[:,:,1]) - y_gt
    closest_dist_gt = tf.sqrt(tf.reduce_sum(tf.square(diff_gt), -1))
    mean_dist_gt = tf.reduce_mean(closest_dist_gt, axis=[1])
    std_dist_gt = tf.math.reduce_std(closest_dist_gt, axis=[1])
    return mean_dist_pred, std_dist_pred, mean_dist_gt, std_dist_gt