"""Code for training CycleGAN."""
from datetime import datetime
import json
import numpy as np
import os
# import random
import imageio
from open3d import *

import click
import tensorflow as tf
# from skimage import io,transform
import sklearn.preprocessing

import losses
import model_ae_pc_adaptive as model
from shapenet_dataset_ae import *
from visu_utils import point_cloud_one_view
import provider

slim = tf.contrib.slim

DECAY_STEP = 100000
DECAY_RATE = 0.7

BN_INIT_DECAY = 0.5
BN_DECAY_CLIP = 0.99
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_DECAY_RATE = 0.5

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3]*np.sin(phi)
    temp = param[3]*np.cos(phi)
    camX = temp * np.cos(theta)    
    camZ = temp * np.sin(theta)        
    cam_pos = np.array([camX, camY, camZ])        

    axisZ = cam_pos.copy()
    axisY = np.array([0,1,0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([axisX, axisY, axisZ])
    cam_mat = sklearn.preprocessing.normalize(cam_mat, axis=1)
    return cam_mat, cam_pos

def rotate_point_cloud(xyz): # rotate point clouds z-upwards
    R_x = np.array([[1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]])
    R_y = np.array([[0, 0, 1],
                    [0, 1, 0],
                    [-1, 0, 0]])
    xyz_rotated = np.dot(xyz, R_x)
    xyz_rotated = np.dot(xyz_rotated, R_y)

    cam_mat, _ = camera_info([45., 25.0,  0, 1.25])
    final_pc = np.dot(xyz_rotated, cam_mat.transpose())
    return final_pc

class AE_PC:

    def __init__(self, output_root_dir, to_restore, to_train,
                 base_lr, max_step, checkpoint_dir, data_list_train, data_list_test):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        if to_train == 0:
            self._output_dir = os.path.join(output_root_dir, "test_ae_pc")
        else:
            self._output_dir = os.path.join(output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._point_clouds_dir = os.path.join(self._output_dir, 'pcs')
        self._num_imgs_to_save = 20
        self._to_restore = to_restore
        self._base_lr = base_lr
        self._max_step = max_step
        self._checkpoint_dir = checkpoint_dir
        self._data_list_train = data_list_train
        self._data_list_test = data_list_test

    def load(self, checkpoint_dir_ae_pc, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        self.model_setup()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver([v for v in tf.all_variables() if 'AE_PC' in v.name])
        chkpt_fname = tf.train.latest_checkpoint(checkpoint_dir_ae_pc)
        saver.restore(sess, chkpt_fname)

    def get_learning_rate(self, batch):
        learning_rate = tf.train.exponential_decay(
                            self._base_lr,  
                            batch * model.BATCH_SIZE,  
                            DECAY_STEP,          
                            DECAY_RATE,          
                            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001) 
        return learning_rate

    def model_setup(self):
        """
        This function sets up the model to train.

        """
        self.input_pc = tf.placeholder(
            tf.float32, [
                None,
                model.PC_POINTS,
                model.PC_CHANNELS
            ], name="input_pc_pl")

        self.input_normals_pc = tf.placeholder(
            tf.float32, [
                None,
                model.PC_POINTS,
                model.PC_CHANNELS
            ], name="input_normals_pl")
            
        self.input_features_pc = tf.placeholder(
            tf.float32, [
                None,
                model.PC_FEATURES
            ], name="features_pc_pl"
        )

        self.surface_area_pc = tf.placeholder(
            tf.float32, [
            None
            ], name="surface_area"
        )

        self.global_step = slim.get_or_create_global_step()

        self.num_fake_inputs = 0

        self.is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
        self.batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = self.get_learning_rate(self.batch)

        inputs = {
            'input_pc': self.input_pc,
            'input_normals_pc': self.input_normals_pc,
            'input_features_pc': self.input_features_pc,
            'bn_decay': None,
            'is_training': self.is_training
        }

        outputs = model.get_outputs(inputs)

        self.out_features_pc = outputs['out_features_pc']
        self.out_pc = outputs['out_pc']
        self.out_pc_from_features = outputs['out_pc_from_features']

    def compute_losses(self):
        """
        In this function we are defining the variables for loss calculations
        and training model.
        """

        emd_loss = losses.emd(self.out_pc, self.input_pc)
        chamfer_loss, d1, d2 = losses.chamfer(self.out_pc, self.input_pc)
        particle_loss, corrected_proj_x, proj_x, queries, normals = losses.particle_loss(self.out_pc, self.input_pc, self.input_normals_pc, self.surface_area_pc)
        uniformity_mean_pred, uniformity_std_pred, uniformity_mean_gt, uniformity_std_gt = losses.evaluate_uformity(self.out_pc, self.input_pc)

        loss = emd_loss + particle_loss

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.model_vars = tf.trainable_variables()

        enc_pc_vars = [var for var in self.model_vars if 'enc_pc' in var.name]
        dec_pc_vars = [var for var in self.model_vars if 'dec_pc' in var.name]

        self.ae_pc_trainer = optimizer.minimize(loss, var_list=[enc_pc_vars, dec_pc_vars], global_step=self.batch)

        for var in self.model_vars:
            print(var.name)

        # Summary variables for tensorboard
        self.emd_loss = emd_loss
        self.cd_loss = chamfer_loss
        self.d1 = d1,
        self.d2 = d2,
        self.uniformity_mean_pred = uniformity_mean_pred
        self.uniformity_std_pred = uniformity_std_pred
        self.uniformity_mean_gt = uniformity_mean_gt
        self.uniformity_std_gt = uniformity_std_gt
        self.corrected_proj_x = corrected_proj_x
        self.proj_x = proj_x,
        self.queries = queries,
        self.normals = normals,
        self.ae_pc_loss =  tf.summary.scalar("total_loss", loss)
        self.ae_pc_loss_emd = tf.summary.scalar("emd_loss", emd_loss)
        self.ae_pc_particle_loss = tf.summary.scalar("particle loss", particle_loss)
        self.ae_pc_lr = tf.summary.scalar('lr', self.learning_rate)

    def save_test_images(self, sess, epoch, data):
        """
        Saves input and output images.

        :param sess: The session.
        :param epoch: Current epoch.
        """

        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)
        
        if not os.path.exists(self._point_clouds_dir):
            os.makedirs(self._point_clouds_dir)

        names_pc = ['inputPC_', 'reconstructedPC_', 'reconstructedPCfromFEAT_', 'pc_sampl_1_', 'pc_sampl_2_', 'pc_sampl_3_']

        with open(os.path.join(self._output_dir, 'epoch_pc_' + str(epoch) + '.html'), 'w') as v_pc_html:
                for i in range(0, self._num_imgs_to_save):
                    print("Saving image {}/{}".format(i, self._num_imgs_to_save))
                    input_pc_, surface_area_pc, _ = data.next_batch()
                    input_pc = input_pc_[:,:,:3]
                    input_normals_pc = input_pc_[:,:,3:]

                    out_features_pc, reconstructed_pc = sess.run([
                        self.out_features_pc,
                        self.out_pc
                    ], feed_dict={
                        self.input_pc: input_pc,
                        self.input_normals_pc: input_normals_pc,
                        self.surface_area_pc: surface_area_pc,
                        self.is_training: False
                    })

                    reconstructed_from_features_pc= sess.run([
                        self.out_pc_from_features,
                    ], feed_dict={
                        self.input_features_pc: out_features_pc,
                        self.is_training: False
                    })[0]

                    input_pc = rotate_point_cloud(input_pc)
                    reconstructed_pc = rotate_point_cloud(reconstructed_pc)
                    reconstructed_from_features_pc = rotate_point_cloud(reconstructed_from_features_pc)

                    tensors_pc = [input_pc, reconstructed_pc, reconstructed_from_features_pc]

                    v_pc_html.write("<br> Object: "+str(i+1)+" <br><br>")

                    for name, tensor in zip(names_pc, tensors_pc):
                        points = tensor[0]
                        pc_name = name + str(epoch) + "_" + str(i) + ".obj"
                        out_filename = os.path.join(self._point_clouds_dir, pc_name)
                        fout = open(out_filename, 'w')
                        for j in range(points.shape[0]):
                                fout.write('v %f %f %f\n' % (points[j,0], points[j,1], points[j,2]))
                        fout.close()

                        # save point cloud as an image
                        image_input_x = point_cloud_one_view(points)
                        image_name = name + str(epoch) + "_" + str(i) + ".jpg"
                        imageio.imsave(os.path.join(self._images_dir, image_name), (image_input_x*255.0).astype(np.uint8))
                        v_pc_html.write("<img src=\"" + os.path.join('imgs', image_name) + "\">")
                    v_pc_html.write("<br>")
            
    def train(self):
        """Training Function."""
        # Load Training and Testing Dataset
        TRAIN_DATASET = ShapeNetDataset(self._data_list_train, batch_size=model.BATCH_SIZE)
        TEST_DATASET = ShapeNetDataset(self._data_list_test, batch_size=1)

        # Build the network
        self.model_setup()

        # Loss function calculations
        self.compute_losses()

        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver = tf.train.Saver()

        max_batches = TRAIN_DATASET.get_num_batches()

        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.allow_soft_placement=True
        config.log_device_placement = False

        with tf.Session(config=config) as sess:
            sess.run(init)

            # Restore the model to run the model from last checkpoint
            if self._to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self._output_dir)

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            # Training Loop
            for epoch in range(sess.run(self.global_step), self._max_step + 1):
                print("In the epoch ", epoch)

                batch_idx = 0
                while TRAIN_DATASET.has_next_batch():
                    print("Processing batch {}/{}".format(batch_idx, max_batches))

                    input_pc_, surface_area_pc, _ = TRAIN_DATASET.next_batch()
                    input_pc_ = provider.shuffle_points(input_pc_)
                    input_pc = input_pc_[:,:,:3]
                    input_normals_pc = input_pc_[:,:,3:]

                    _, summary_loss_total, summary_loss_emd, summary_particle_loss, summary_lr = sess.run(
                        [self.ae_pc_trainer,
                         self.ae_pc_loss,
                         self.ae_pc_loss_emd,
                         self.ae_pc_particle_loss,
                         self.ae_pc_lr
                         ],
                        feed_dict={
                            self.input_pc: input_pc,
                            self.input_normals_pc: input_normals_pc,
                            self.surface_area_pc: surface_area_pc,
                            self.is_training: True
                        }
                    )
                    writer.add_summary(summary_loss_total, epoch * max_batches + batch_idx)
                    writer.add_summary(summary_loss_emd, epoch * max_batches + batch_idx)
                    writer.add_summary(summary_particle_loss, epoch * max_batches + batch_idx)
                    writer.add_summary(summary_lr, epoch * max_batches + batch_idx)

                    writer.flush()
                    batch_idx += 1
                
                saver.save(sess, os.path.join(self._output_dir, "ae_pc"))

                batch_idx = 0
                self.save_test_images(sess, epoch, TEST_DATASET)
                TEST_DATASET.start_from_the_first_batch_again()
                TRAIN_DATASET.reset_one_view()
                sess.run(tf.assign(self.global_step, epoch + 1))

            writer.add_graph(sess.graph)

    def f_score(self, label, predict, dist_label, dist_pred, threshold):
        num_label = label.shape[0]
        num_predict = predict.shape[0]

        f_scores = []
        for i in range(len(threshold)):
            num = len(np.where(dist_label <= threshold[i])[0])
            recall = 100.0 * num / num_label
            num = len(np.where(dist_pred <= threshold[i])[0])
            precision = 100.0 * num / num_predict

            f_scores.append((2*precision*recall)/(precision+recall+1e-8))
        return np.array(f_scores)

    def evaluate_test_emd_and_uniformity_loss(self, sess, data):
        batch_idx = 0
        max_batches = data.get_num_batches()

        class_name = {	'02691156':'plane',
				'02828884':'bench',
				'02933112':'cabinet',
				'02958343':'car',
				'03001627':'chair',
				'03211117':'monitor',
				'03636649':'lamp',
				'03691459':'speaker',
				'04090263':'firearm',
				'04256520':'couch',
				'04379243':'table',
				'04401088':'cellphone',
				'04530566':'watercraft'
				}
        model_number = {i:0 for i in class_name}
        sum_f = {i:0 for i in class_name}
        sum_cd = {i:0 for i in class_name}
        sum_emd = {i:0 for i in class_name}
        sum_uniformity_mean_pred = {i:0 for i in class_name}
        sum_uniformity_std_pred = {i:0 for i in class_name}
        sum_uniformity_mean_gt = {i:0 for i in class_name}
        sum_uniformity_std_gt = {i:0 for i in class_name}

        if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

        while data.has_next_batch():
            if batch_idx % 100 == 0:
                print("Processing batch {}/{}".format(batch_idx, max_batches))
            input_pc_, surface_area_pc, model_id = data.next_batch()
            input_pc = input_pc_[:,:,:3]
            input_normals_pc = input_pc_[:,:,3:]

            emd_loss, cd_loss, d1, d2, out_pc, uniformity_mean_pred, uniformity_std_pred, uniformity_mean_gt, uniformity_std_gt = sess.run(
                [self.emd_loss, 
                 self.cd_loss,
                 self.d1,
                 self.d2,
                 self.out_pc,
                 self.uniformity_mean_pred,
                 self.uniformity_std_pred,
                 self.uniformity_mean_gt,
                 self.uniformity_std_gt],
                feed_dict={
                    self.input_pc: input_pc,
                    self.input_normals_pc: input_normals_pc,
                    self.surface_area_pc: surface_area_pc,
                    self.is_training: False
                }
            )

            class_id = model_id[0]
            model_number[class_id] += 1.0
            sum_emd[class_id] += emd_loss
            sum_cd[class_id] += cd_loss
            sum_f[class_id] += self.f_score(input_pc[0],out_pc[0],d1[0],d2[0],[0.0002, 0.0004])
            sum_uniformity_mean_pred[class_id] += uniformity_mean_pred
            sum_uniformity_std_pred[class_id] += uniformity_std_pred
            sum_uniformity_mean_gt[class_id] += uniformity_mean_gt
            sum_uniformity_std_gt[class_id] += uniformity_std_gt
            batch_idx += 1
        
        cd_sum = 0.0
        emd_sum = 0.0
        f_sum = 0.0
        uniformity_mean_pred_sum = 0.0
        uniformity_std_pred_sum = 0.0
        uniformity_mean_gt_sum = 0.0
        uniformity_std_gt_sum = 0.0

        log = open(os.path.join(self._output_dir, 'record_evaluation.txt'), 'a')

        for item in model_number:
            number = model_number[item] + 1e-8
            emd = (sum_emd[item] / number) * 0.01
            cd = (sum_cd[item] / number) * 1000
            f = sum_f[item] / number
            uniform_mean_pred = sum_uniformity_mean_pred[item] / number
            uniform_std_pred = sum_uniformity_std_pred[item] / number
            uniform_mean_gt = sum_uniformity_mean_gt[item] / number
            uniform_std_gt = sum_uniformity_std_gt[item] / number
            cd_sum += cd
            f_sum += f
            emd_sum += emd
            uniformity_mean_pred_sum += uniform_mean_pred
            uniformity_std_pred_sum += uniform_std_pred
            uniformity_mean_gt_sum += uniform_mean_gt
            uniformity_std_gt_sum += uniform_std_gt

            print(class_name[item], int(number), f, cd, emd, uniform_mean_pred, uniform_std_pred, uniform_mean_gt, uniform_std_gt)
            log.write(str(class_name[item]) + ', ' + str(int(number)) + ', ' + str(f) + ', ' + str(cd) + ', ' + str(emd) \
                + ', ' + str(uniform_mean_pred) + ', ' + str(uniform_std_pred) + ', ' + str(uniform_mean_gt) + ', ' + str(uniform_std_gt) + '\n')

        # print('mean: ', f_sum/13.0, cd_sum/13.0 , emd_sum/13.0, \
        #     uniformity_mean_pred_sum/13.0, uniformity_std_pred_sum/13.0, uniformity_mean_gt_sum/13.0, uniformity_std_gt_sum/13.0)
        log.write('mean: ' + str(f_sum/13.0) + ', ' + str(cd_sum/13.0) + ', ' + str(emd_sum/13.0) \
            + ', ' + str(uniformity_mean_pred_sum/13.0) + ', ' + str(uniformity_std_pred_sum/13.0) \
            + ', ' + str(uniformity_mean_gt_sum/13.0) + ', ' + str(uniformity_std_gt_sum/13.0) + '\n')
        log.close()

    def test(self):
        """Test Function."""

        TEST_DATASET = ShapeNetDataset(self._data_list_test, batch_size=1)

        self.model_setup()
        self.compute_losses()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.allow_soft_placement=True
        config.log_device_placement = False

        with tf.Session(config=config) as sess:
            sess.run(init)

            chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
            saver.restore(sess, chkpt_fname)

            self.evaluate_test_emd_and_uniformity_loss(sess, TEST_DATASET)
            TEST_DATASET.start_from_the_first_batch_again()
            self.save_test_images(sess, "test", TEST_DATASET)


@click.command()
@click.option('--to_train',
              type=click.INT,
              default=True,
              help='Whether it is train or false.')
@click.option('--log_dir',
              type=click.STRING,
              default=None,
              help='Where the data is logged to.')
@click.option('--config_filename',
              type=click.STRING,
              default='train',
              help='The name of the configuration file.')
@click.option('--checkpoint_dir',
              type=click.STRING,
              default='',
              help='The name of the train/test split.')
def main(to_train, log_dir, config_filename, checkpoint_dir):
    """

    :param to_train: Specify whether it is training or testing. 1: training; 2:
     resuming from latest checkpoint; 0: testing.
    :param log_dir: The root dir to save checkpoints and imgs. The actual dir
    is the root dir appended by the folder with the name timestamp.
    :param config_filename: The configuration file.
    :param checkpoint_dir: The directory that saves the latest checkpoint. It
    only takes effect when to_train == 2.
    :param skip: A boolean indicating whether to add skip connection between
    input and output.
    """
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    with open(config_filename) as config_file:
        config = json.load(config_file)

    to_restore = (to_train == 2)
    base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002
    max_step = int(config['max_step']) if 'max_step' in config else 200
    data_list_train = str(config['data_list_train'])
    data_list_test = str(config['data_list_test'])

    ae_model = AE_PC(log_dir, to_restore, to_train, base_lr, max_step, checkpoint_dir, data_list_train, data_list_test)

    if to_train > 0:
        ae_model.train()
    else:
        ae_model.test()


if __name__ == '__main__':
    main()
