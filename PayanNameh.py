# -*- coding: utf-8 -*-
from __future__ import division

import binascii
import os
import time
from math import sqrt

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from ops import *
from utils import *


class EBGAN(object):
    model_name = "EBGAN"  # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim  # dimension of noise-vector
            self.c_dim = 1

            # EBGAN Parameter
            self.pt_loss_weight = 0.1
            self.margin = max(1, self.batch_size / 64.)  # margin for loss function
            # usually margin of 1 is enough, but for large batch size it must be larger than 1

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    # borrowed from https://github.com/shekkizh/EBGAN.tensorflow/blob/master/EBGAN/Faces_EBGAN.py
    def pullaway_loss(self, embeddings):
        """
        Pull Away loss calculation
        :param embeddings: The embeddings to be orthogonalized for varied faces. Shape [batch_size, embeddings_dim]
        :return: pull away term loss
        """
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        similarity = tf.matmul(
            normalized_embeddings, normalized_embeddings, transpose_b=True)
        batch_size = tf.cast(tf.shape(embeddings)[0], tf.float32)
        pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return pt_loss

    def discriminator(self, x, is_training=True, reuse=False):
        # It must be Auto-Encoder style architecture
        with tf.compat.v1.variable_scope("discriminator", reuse=reuse):
            # encode
            net1 = tf.nn.relu(conv2d(x, 64, 5, 5, 2, 2, name='d_conv1'))
            net2 = tf.reshape(net1, [self.batch_size, -1])
            code = (linear(net2, 64, scope='d_fc2'))  # bn and relu are excluded since code is used in pullaway_loss

            # decode
            net3 = tf.nn.relu(bn(linear(code, 64 * 14 * 14, scope='d_fc4'), is_training=is_training, scope='d_bn4'))
            net4 = tf.reshape(net3, [self.batch_size, 14, 14, 64])
            img_out = tf.nn.sigmoid(deconv2d(net4, [self.batch_size, 28, 28, 1], 5, 5, 2, 2, name='d_dc2'))

            # get correctness probability
            net03 = tf.nn.relu(bn(code, is_training=is_training, scope='d_bn3'))
            prob_logit = linear(net03, 1, scope='d_fcp2')
            prob = tf.nn.sigmoid(prob_logit)

            # recon loss
            recon_error = tf.sqrt(2 * tf.nn.l2_loss(img_out - x)) / self.batch_size
            return img_out, recon_error, code, prob, prob_logit

    def generator(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):
            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 14, 14, 64], 5, 5, 2, 2, name='g_dc3'), is_training=is_training,
                   scope='g_bn3'))

            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 5, 5, 2, 2, name='g_dc4'))

            return out

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.compat.v1.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises
        self.z = tf.compat.v1.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        # output of D for real images
        D_real_img, D_real_err, D_real_code, D_real_prob, D_real_logits = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, is_training=True, reuse=False)
        D_fake_img, D_fake_err, D_fake_code, D_fake_prob, D_fake_logits = self.discriminator(G, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real_prob = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_prob)))
        d_loss_fake_prob = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_prob)))

        self.d_loss = D_real_err + tf.maximum(self.margin - D_fake_err, 0) + d_loss_real_prob + d_loss_fake_prob

        # get loss for generator
        self.g_loss = D_fake_err + self.pt_loss_weight * self.pullaway_loss(D_fake_code) + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_prob)))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.compat.v1.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.compat.v1.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.compat.v1.train.AdamOptimizer(self.learning_rate*10, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)
        self.real_images, self.real_error, self.real_code, self.real_prob, _ = self.discriminator(self.inputs, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_error_real", D_real_err)
        d_loss_fake_sum = tf.summary.scalar("d_error_fake", D_fake_err)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.compat.v1.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.compat.v1.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):

        # initialize all variables
        tf.compat.v1.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        # saver to save model
        self.saver = tf.compat.v1.train.Saver()

        # summary writer
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                       feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                       feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(
                                    self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)
            self.test_discriminator(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    def get_mse_losses(self, data):
        losses = []
        for idx in range(0, self.num_batches):
            print("MSE Loss Batch Number:" + str(idx))
            batch_images = data[idx * self.batch_size:(idx + 1) * self.batch_size]
            mse_loss = self.sess.run(self.real_error, feed_dict={self.inputs: batch_images})
            losses.append(mse_loss)
            # print("MSE Loss probability: " + str(mse_loss))
        return losses

    def get_corr_probs(self, data):
        corr_nums = []
        for idx in range(0, self.num_batches):
            print("Correctness Batch Number:" + str(idx))
            batch_images = data[idx * self.batch_size:(idx + 1) * self.batch_size]
            corr = self.sess.run(self.real_prob, feed_dict={self.inputs: batch_images})
            sum = 0
            for idy in range(0, self.batch_size):
                sum += corr[idy][0]
            corr_nums.append(sum / self.batch_size)
        return corr_nums

    def plot_batches(self, my_arr, my_arr2, label, threshold=None):
        indices = np.arange(self.num_batches)
        if threshold is None:
            plt.plot(indices, my_arr, 'g', my_arr2, 'r')
        else:
            threshold_arr = []
            for i in range(0, self.num_batches):
                threshold_arr.append(threshold)
            plt.plot(indices, my_arr, 'g', my_arr2, 'r', threshold_arr, 'b')
            plt.axis([0, 1093, 0, 1])
        plt.xlabel('Batch Number')
        plt.ylabel(label)
        plt.savefig(label + '.png')
        plt.show()

    def test_mse_losses(self):
        fake_data_x, _ = load_mnist('fashion-mnist')
        real_losses = self.get_mse_losses(self.data_X)
        fake_losses = self.get_mse_losses(fake_data_x)
        self.plot_batches(real_losses, fake_losses, 'MSE Loss')

    def test_corr_probs(self):
        fake_data_x, _ = load_mnist('fashion-mnist')
        real_probs = self.get_corr_probs(self.data_X)
        fake_probs = self.get_corr_probs(fake_data_x)
        threshold = 0.5
        false_alarm = 0
        for i in range(0, self.num_batches):
            if fake_probs[i] > threshold:
                false_alarm += 1
            if real_probs[i] < threshold:
                false_alarm += 1
        print("False Alarm: " + str(false_alarm))
        self.plot_batches(real_probs, fake_probs, 'Correctness', threshold)

    def plot_false_alarms(self):
        objects = ('Code Size = 16', 'Code Size = 32', 'Code Size = 64')
        y_pos = np.arange(len(objects))
        performance = [1106, 21, 0.0000001]

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('False')
        plt.title('False Alarms')
        plt.savefig('False Alarms.png')
        plt.show()

    def test_discriminator(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        # get batch data
        print("Number of Batches: " + str(self.num_batches))

        idx = 0
        batch_images = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]

        save_images(batch_images[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_input_discreminator_test_all_classes.png')
        save_images(batch_images[:1, :, :, :], [1, 1],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_input_one_img_discreminator_test_all_classes.png')

        samples = self.sess.run(self.real_images, feed_dict={self.inputs: batch_images})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_' + str(epoch) + '_output_discreminator_test_all_classes.png')
        save_images(samples[:1, :, :, :], [1, 1],
                    check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_' + str(epoch) + '_output_one_img_discreminator_test_all_classes.png')

        # # Load image as grayscale
        # image = cv2.imread(check_folder(
        #                 self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_' + str(epoch) + '_output_discreminator_test_all_classes.png', cv2.IMREAD_GRAYSCALE)
        #
        # # Enhance image
        # image_enhanced = cv2.equalizeHist(image)
        # image_enhanced = cv2.fastNlMeansDenoising(image_enhanced)
        #
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # image_enhanced = cv2.filter2D(image_enhanced, -1, kernel)
        #
        # cv2.imwrite(check_folder(
        #                 self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_' + str(epoch) + '_denoised_output_discreminator_test_all_classes.png', image_enhanced)

    def watermark(self, batch_images, scale_number, bit_repeat, key):
        normalized_images = np.array((batch_images + 1.) / 2.)
        scaled_images = (normalized_images * scale_number).astype(int)
        binkey_length = len(key) * 8
        key_binary = format(int(binascii.hexlify(key), 16), "0" + str(binkey_length) + "b")
        print("key in binary:" + key_binary)

        for h in range(len(scaled_images)):
            kindex = 0
            for i in range(28):
                for j in range(int(28 / bit_repeat)):
                    for k in range(bit_repeat):
                        col = (j * bit_repeat) + k
                        if kindex < binkey_length:
                            if int(key_binary[kindex]) == 1 and scaled_images[h][i][col][0] % 2 == 0:
                                scaled_images[h][i][col][0] += 1
                            elif int(key_binary[kindex]) == 0 and scaled_images[h][i][col][0] % 2 == 1:
                                scaled_images[h][i][col][0] -= 1
                        elif scaled_images[h][i][col][0] % 2 == 1:
                            scaled_images[h][i][col][0] -= 1
                    kindex += 1

        unscaled_images = scaled_images.astype(float) / scale_number
        return (unscaled_images * 2.) - 1.

    def dewatermark(self, samples, scale_number, bit_repeat, binkey_length):
        normalized_samples = np.array((samples + 1.) / 2.)
        scaled_samples = (normalized_samples * scale_number).astype(int)
        raw_samples_key = scaled_samples % 2
        for h in range(len(raw_samples_key)):
            mykey_binary = ""
            for i in range(28):
                for j in range(int(28 / bit_repeat)):
                    count1 = 0
                    for k in range(bit_repeat):
                        col = (j * bit_repeat) + k
                        if len(mykey_binary) < binkey_length and raw_samples_key[h][i][col][0] == 1:
                            count1 += 1
                    if len(mykey_binary) < binkey_length:
                        if count1 >= (bit_repeat / 2):
                            mykey_binary += "1"
                        else:
                            mykey_binary += "0"
            print(mykey_binary)


    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
