""" Joint Latent Space

Author: A. Komarichev
"""
from datetime import datetime
import json
import numpy as np
import os
import random
import imageio
from open3d import *

import click
import tensorflow as tf
from skimage import io,transform

import losses
import model_ae_img as model
from shapenet_dataset_img import *
from visu_utils import point_cloud_three_views, point_cloud_one_view
import provider

slim = tf.contrib.slim

os.environ["CUDA_VISIBLE_DEVICES"]="1"

class AE_IMG:
    """The CycleGAN module."""

    def __init__(self, _lambda, output_root_dir, to_restore, to_train,
                 base_lr, max_step, checkpoint_dir, data_list_train, data_list_test):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        if to_train == 0:
            self._output_dir = os.path.join(output_root_dir, "test_ae_img")
        else:
            self._output_dir = os.path.join(output_root_dir, current_time)
        self._lambda = _lambda
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._point_clouds_dir = os.path.join(self._output_dir, 'pcs')
        self._num_imgs_to_save = 20
        self._to_restore = to_restore
        self._base_lr = base_lr
        self._max_step = max_step
        self._checkpoint_dir = checkpoint_dir
        self._data_list_train = data_list_train
        self._data_list_test = data_list_test

    def load(self, checkpoint_dir_ae_img, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        self.model_setup()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver([v for v in tf.all_variables() if 'AE_IMG' in v.name])
        chkpt_fname = tf.train.latest_checkpoint(checkpoint_dir_ae_img)
        saver.restore(sess, chkpt_fname)
        print("Model AE on images restored from file: %s" % chkpt_fname)

    def model_setup(self):
        """
        This function sets up the model to train.

        self.input_A/self.input_B -> Set of training images.
        self.fake_A/self.fake_B -> Generated images by corresponding generator
        of input_A and input_B
        self.lr -> Learning rate variable
        self.cyc_A/ self.cyc_B -> Images generated after feeding
        self.fake_A/self.fake_B to corresponding generator.
        This is use to calculate cyclic loss
        """
        self.input_img = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS
            ], name="input_img_pl")
            
        self.input_features_img = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_FEATURES
            ], name="features_pc_pl"
        )

        self.global_step = slim.get_or_create_global_step()

        self.num_fake_inputs = 0

        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")
        self.is_training = tf.placeholder(tf.bool, shape=[], name="is_training")

        inputs = {
            'input_img': self.input_img,
            'input_features_img': self.input_features_img,
            'is_training': self.is_training
        }

        outputs = model.get_outputs(inputs)

        self.out_features_img = outputs['out_features_img']
        self.out_img = outputs['out_img']
        self.out_img_from_features = outputs['out_img_from_features']

    def compute_losses(self):
        """
        In this function we are defining the variables for loss calculations
        and training model.
        """

        abs_loss = self._lambda * losses.cycle_consistency_loss(real_images=self.input_img, generated_images=self.out_img)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.model_vars = tf.trainable_variables()

        enc_img_vars = [var for var in self.model_vars if 'enc_img' in var.name]
        dec_img_vars = [var for var in self.model_vars if 'dec_img' in var.name]

        self.ae_img_trainer = optimizer.minimize(abs_loss, var_list=[enc_img_vars, dec_img_vars])

        for var in self.model_vars:
            print(var.name)

        # Summary variables for tensorboard
        self.ae_img_loss_summ = tf.summary.scalar("abs_loss", abs_loss)

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

        names_img = ['inputIMG_', 'reconstructedIMG_', 'reconstructedIMGfromFEAT_']

        with open(os.path.join(self._output_dir, 'epoch_img_' + str(epoch) + '.html'), 'w') as v_img_html:
                for i in range(0, self._num_imgs_to_save):
                    print("Saving image {}/{}".format(i, self._num_imgs_to_save))
                    input_img = data.next_batch()

                    out_features_img, reconstructed_img = sess.run([
                        self.out_features_img,
                        self.out_img,
                    ], feed_dict={
                        self.input_img: input_img,
                        self.is_training: False
                    })

                    reconstructed_from_features_img = sess.run([
                        self.out_img_from_features,
                    ], feed_dict={
                        self.input_features_img: out_features_img,
                        self.is_training: False
                    })[0]

                    tensors_img = [input_img, reconstructed_img, reconstructed_from_features_img]

                    v_img_html.write("<br> Object: "+str(i+1)+" <br><br>")

                    for name, tensor in zip(names_img, tensors_img):
                        image_name = name + str(epoch) + "_" + str(i) + ".jpg"
                        imageio.imsave(os.path.join(self._images_dir, image_name), ((tensor[0] + 1) * 127.5).astype(np.uint8))
                        v_img_html.write("<img src=\"" + os.path.join('imgs', image_name) + "\">")
                    v_img_html.write("<br>")
            
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

        # max_images = cyclegan_datasets.DATASET_TO_SIZES[self._dataset_name]
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
            curr_lr = self._base_lr
            for epoch in range(sess.run(self.global_step), self._max_step + 1):
                # Dealing with the learning rate as per the epoch number
                if epoch < 200:
                    curr_lr = self._base_lr
                else:
                    curr_lr = self._base_lr - \
                        self._base_lr * (epoch - 200) / 201

                print("In the epoch ", epoch, ', lr = ', curr_lr)

                # for i in range(0, max_images):
                batch_idx = 0
                while TRAIN_DATASET.has_next_batch():
                    print("Processing batch {}/{}".format(batch_idx, max_batches))

                    input_img = TRAIN_DATASET.next_batch()

                    # Optimizing AE
                    _, summary_str = sess.run(
                        [self.ae_img_trainer,
                         self.ae_img_loss_summ],
                        feed_dict={
                            self.input_img: input_img,
                            self.learning_rate: curr_lr,
                            self.is_training: True
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_batches + batch_idx)

                    writer.flush()
                    batch_idx += 1
                
                batch_idx = 0
                self.save_test_images(sess, epoch, TEST_DATASET)
                saver.save(sess, os.path.join(self._output_dir, "ae_img"))
                TEST_DATASET.start_from_the_first_batch_again()
                TRAIN_DATASET.reset_one_view()
                sess.run(tf.assign(self.global_step, epoch + 1))

            writer.add_graph(sess.graph)

    def test(self):
        """Test Function."""
        print("Testing the results")

        TEST_DATASET = ShapeNetDataset("Data/test_list_cellphone_all_24_views.txt", batch_size=model.BATCH_SIZE, split='test')

        self.model_setup()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        max_images = TEST_DATASET.get_num_batches_one_view()

        print("# number of images: ", max_images)

        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.allow_soft_placement=True
        config.log_device_placement = False

        with tf.Session(config=config) as sess:
            sess.run(init)

            chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
            saver.restore(sess, chkpt_fname)


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

    _lambda = float(config['_lambda']) if '_lambda' in config else 10.0

    to_restore = (to_train == 2)
    base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002
    max_step = int(config['max_step']) if 'max_step' in config else 200
    data_list_train = str(config['data_list_train'])
    data_list_test = str(config['data_list_test'])

    ae_model = AE_IMG(_lambda, log_dir, to_restore, to_train, base_lr, max_step, checkpoint_dir, data_list_train, data_list_test)

    if to_train > 0:
        ae_model.train()
    else:
        ae_model.test()


if __name__ == '__main__':
    main()
