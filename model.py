#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math

class PUNet(object): # variational auto-encoder model
    def __init__(self, img_h, img_w, img_channel, code_dim, base_channel, class_channel):
        self.INPUT_IMAGE_SIZE_W = img_w
        self.INPUT_IMAGE_SIZE_H = img_h
        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE1 = 7
        self.CONV_FILTER_SIZE2 = 3
        self.CODE_DIMENTION = code_dim
        self.CONV_STRIDE1 = 2
        self.CONV_STRIDE2 = 2
        self.INPUT_CHANNEL = img_channel
        self.CLASS_CHANNEL = class_channel
        self.BASE_CHANNEL = base_channel
        # self.LATENT_DIM = 6
        self.SEED = 2018
        self.KL_BETA = 1.
        print("self.INPUT_IMAGE_SIZE_W = ", self.INPUT_IMAGE_SIZE_W)
        print("self.INPUT_IMAGE_SIZE_H = ", self.INPUT_IMAGE_SIZE_H)
        print("self.CONCATENATE_AXIS = ", self.CONCATENATE_AXIS)
        print("self.CONV_FILTER_SIZE1 = ", self.CONV_FILTER_SIZE1)
        print("self.CONV_FILTER_SIZE2 = ", self.CONV_FILTER_SIZE2)
        print("self.CODE_DIMENTION = ", self.CODE_DIMENTION)
        print("self.CONV_STRIDE1 = ", self.CONV_STRIDE1)
        print("self.CONV_STRIDE2 = ", self.CONV_STRIDE2)
        print("self.INPUT_CHANNEL = ", self.INPUT_CHANNEL)
        print("self.BASE_CHANNEL = ", self.BASE_CHANNEL)

    def cal_input_num(self, input_num):
        stddev = math.sqrt(2 / (input_num))
        return stddev

    def conv2d(self, input, in_channel, out_channel, k_size, stride, seed):
        stdd = self.cal_input_num(in_channel * k_size * k_size)
        w = tf.get_variable('w', [k_size, k_size, in_channel, out_channel],
                            initializer=tf.random_normal_initializer
                            (mean=0.0, stddev=stdd, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding="SAME", name='conv') + b
        return conv

    def conv2d_transpose(self, input, in_channel, out_channel, k_size, stride, shape, seed):
        stdd = self.cal_input_num(in_channel * k_size * k_size)
        w = tf.get_variable('w', [k_size, k_size, out_channel, in_channel],
                            initializer=tf.random_normal_initializer
                            (mean=0.0, stddev=stdd, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        # out_shape = tf.stack(
        #                 [tf.shape(input)[0], tf.shape(input)[1] * 2, tf.shape(input)[2] * 2, tf.constant(out_channel)])
        deconv = tf.nn.conv2d_transpose(input, w, output_shape=shape, strides=[1, stride, stride, 1],
                                        padding="SAME") + b
        return deconv

    def batch_norm(self, input):
        shape = input.get_shape().as_list()
        n_out = shape[-1]
        scale = tf.get_variable('scale', [n_out], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer(0.0))
        batch_mean, batch_var = tf.nn.moments(input, [0])
        bn = tf.nn.batch_normalization(input, batch_mean, batch_var, beta, scale, 0.0001, name='batch_norm')
        return bn

    def fully_connect(self, input, in_num, out_num, seed):
        w = tf.get_variable('w', [in_num, out_num], initializer=tf.random_normal_initializer
        (mean=0.0, stddev=0.02, seed=seed), dtype=tf.float32)
        b = tf.get_variable('b', [out_num], initializer=tf.constant_initializer(0.0))
        fc = tf.matmul(input, w, name='fc') + b
        return fc

    def priorNet(self, input_data):
        with tf.variable_scope("PriorNet"):
            with tf.variable_scope("priornet_1"):  # 128x128 -> 64x64
                self.input_data = input_data
                conv1 = self.conv2d(input_data, self.INPUT_CHANNEL, self.BASE_CHANNEL, self.CONV_FILTER_SIZE1,
                                    1, self.SEED)
                conv1 = self.batch_norm(conv1)
                conv1_relu = tf.nn.relu(conv1)
                conv1_relu = tf.nn.max_pool(conv1_relu, [1,2,2,1], [1,2,2,1], padding='SAME')

            with tf.variable_scope("priornet_2"):  # 64x64 -> 32x32
                conv2 = self.conv2d(conv1_relu, self.BASE_CHANNEL, self.BASE_CHANNEL * 2, self.CONV_FILTER_SIZE2,
                                    1, self.SEED)
                conv2 = self.batch_norm(conv2)
                conv2_relu = tf.nn.relu(conv2)
                conv2_relu = tf.nn.max_pool(conv2_relu, [1,2,2,1], [1,2,2,1], padding='SAME')

            with tf.variable_scope("priornet_3"):  # 32x32 -> 16x16
                conv3 = self.conv2d(conv2_relu, self.BASE_CHANNEL * 2, self.BASE_CHANNEL * 4, self.CONV_FILTER_SIZE2,
                                    1, self.SEED)
                conv3 = self.batch_norm(conv3)
                conv3_relu = tf.nn.relu(conv3)
                conv3_relu = tf.nn.max_pool(conv3_relu, [1,2,2,1], [1,2,2,1], padding='SAME')

            with tf.variable_scope("priornet_4"):  # 16x16 -> 8x8
                conv4 = self.conv2d(conv3_relu, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 8, self.CONV_FILTER_SIZE2,
                                    1, self.SEED)
                conv4 = self.batch_norm(conv4)
                conv4_relu = tf.nn.relu(conv4)
                conv4_relu = tf.nn.max_pool(conv4_relu, [1,2,2,1], [1,2,2,1], padding='SAME')

            with tf.variable_scope("priornet_mean_1"):  # 8x8 -> 4x4
                convME = self.conv2d(conv4_relu, self.BASE_CHANNEL * 8, self.BASE_CHANNEL * 8, self.CONV_FILTER_SIZE2,
                                    1, self.SEED)
                convME_sig = tf.nn.sigmoid(convME)
                convME_sig = tf.nn.max_pool(convME_sig, [1,2,2,1], [1,2,2,1], padding="SAME")

            with tf.variable_scope("priornet_mean_2"):  # 4x4x256 -> z-dim
                # reshape
                fcME_node = tf.shape(convME_sig)[1] * tf.shape(convME_sig)[2] * tf.shape(convME_sig)[3]
                fcME_node_list = convME_sig.get_shape().as_list()
                # print("fcME_node_list, ", fcME_node_list)
                fcME_node_scalar = fcME_node_list[1] * fcME_node_list[2] * fcME_node_list[3]
                # print("fcME_node_scalar, ", fcME_node_scalar)
                convME_re = tf.reshape(convME_sig, [tf.shape(convME_sig)[0], fcME_node])
                mean = self.fully_connect(convME_re, fcME_node_scalar, self.CODE_DIMENTION, self.SEED)
                # print("fcME_node.get_shape(), ", fcME_node.get_shape())
                # print("convME_re.get_shape(), ", convME_re.get_shape())
                # print("mean.get_shape(), ", mean.get_shape())
            with tf.variable_scope("priornet_var_1"):  # 8x8 -> 4x4
                convVA = self.conv2d(conv4_relu, self.BASE_CHANNEL * 8, self.BASE_CHANNEL * 8, self.CONV_FILTER_SIZE2,
                                     1, self.SEED)
                convVA_sig = tf.nn.sigmoid(convVA)
                convVA_sig = tf.nn.max_pool(convVA_sig, [1,2,2,1], [1,2,2,1], padding="SAME")

            with tf.variable_scope("priornet_var_2"):  # 4x4x256 -> z-dim
                fcVA_node = tf.shape(convVA_sig)[1] * tf.shape(convVA_sig)[2] * tf.shape(convVA_sig)[3]
                fcVA_node_list = convVA_sig.get_shape().as_list()
                fcVA_node_scalar = fcVA_node_list[1] * fcVA_node_list[2] * fcVA_node_list[3]
                convVA_re = tf.reshape(convVA_sig, [tf.shape(convVA_sig)[0], fcVA_node])
                log_var = self.fully_connect(convVA_re, fcVA_node_scalar, self.CODE_DIMENTION, self.SEED)
                # self.var = tf.exp(var)
            return mean, log_var
        
        
    def posteriorNet(self, img, mask):
        with tf.variable_scope("Posterior"):
            with tf.variable_scope("posterior_1"):  # 128x128 -> 64x64
                con1 = tf.concat([img, mask], axis=3)
                # self.input_data = input_data
                conv1 = self.conv2d(con1, self.INPUT_CHANNEL + self.CLASS_CHANNEL, self.BASE_CHANNEL, self.CONV_FILTER_SIZE1,
                                    1, self.SEED)
                conv1 = self.batch_norm(conv1)
                conv1_relu = tf.nn.relu(conv1)
                conv1_relu = tf.nn.max_pool(conv1_relu, [1,2,2,1], [1,2,2,1], padding='SAME')

            with tf.variable_scope("posterior_2"):  # 64x64 -> 32x32
                conv2 = self.conv2d(conv1_relu, self.BASE_CHANNEL, self.BASE_CHANNEL * 2, self.CONV_FILTER_SIZE2,
                                    1, self.SEED)
                conv2 = self.batch_norm(conv2)
                conv2_relu = tf.nn.relu(conv2)
                conv2_relu = tf.nn.max_pool(conv2_relu, [1,2,2,1], [1,2,2,1], padding='SAME')

            with tf.variable_scope("posterior_3"):  # 32x32 -> 16x16
                conv3 = self.conv2d(conv2_relu, self.BASE_CHANNEL * 2, self.BASE_CHANNEL * 4, self.CONV_FILTER_SIZE2,
                                    1, self.SEED)
                conv3 = self.batch_norm(conv3)
                conv3_relu = tf.nn.relu(conv3)
                conv3_relu = tf.nn.max_pool(conv3_relu, [1,2,2,1], [1,2,2,1], padding='SAME')

            with tf.variable_scope("posterior_4"):  # 16x16 -> 8x8
                conv4 = self.conv2d(conv3_relu, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 8, self.CONV_FILTER_SIZE2,
                                    1, self.SEED)
                conv4 = self.batch_norm(conv4)
                conv4_relu = tf.nn.relu(conv4)
                conv4_relu = tf.nn.max_pool(conv4_relu, [1,2,2,1], [1,2,2,1], padding='SAME')

            with tf.variable_scope("posterior_mean_1"):  # 8x8 -> 4x4
                convME = self.conv2d(conv4_relu, self.BASE_CHANNEL * 8, self.BASE_CHANNEL * 8, self.CONV_FILTER_SIZE2,
                                    1, self.SEED)
                convME_sig = tf.nn.sigmoid(convME)
                convME_sig = tf.nn.max_pool(convME_sig, [1,2,2,1], [1,2,2,1], padding="SAME")

            with tf.variable_scope("posterior_mean_2"):  # 4x4x256 -> z-dim
                # reshape
                fcME_node = tf.shape(convME_sig)[1] * tf.shape(convME_sig)[2] * tf.shape(convME_sig)[3]
                fcME_node_list = convME_sig.get_shape().as_list()
                # print("fcME_node_list, ", fcME_node_list)
                fcME_node_scalar = fcME_node_list[1] * fcME_node_list[2] * fcME_node_list[3]
                # print("fcME_node_scalar, ", fcME_node_scalar)
                convME_re = tf.reshape(convME_sig, [tf.shape(convME_sig)[0], fcME_node])
                mean = self.fully_connect(convME_re, fcME_node_scalar, self.CODE_DIMENTION, self.SEED)
                # print("fcME_node.get_shape(), ", fcME_node.get_shape())
                # print("convME_re.get_shape(), ", convME_re.get_shape())
                # print("mean.get_shape(), ", mean.get_shape())
            with tf.variable_scope("posterior_var_1"):  # 8x8 -> 4x4
                convVA = self.conv2d(conv4_relu, self.BASE_CHANNEL * 8, self.BASE_CHANNEL * 8, self.CONV_FILTER_SIZE2,
                                     1, self.SEED)
                convVA_sig = tf.nn.sigmoid(convVA)
                convVA_sig = tf.nn.max_pool(convVA_sig, [1,2,2,1], [1,2,2,1], padding="SAME")

            with tf.variable_scope("posterior_var_2"):  # 4x4x256 -> z-dim
                fcVA_node = tf.shape(convVA_sig)[1] * tf.shape(convVA_sig)[2] * tf.shape(convVA_sig)[3]
                fcVA_node_list = convVA_sig.get_shape().as_list()
                fcVA_node_scalar = fcVA_node_list[1] * fcVA_node_list[2] * fcVA_node_list[3]
                convVA_re = tf.reshape(convVA_sig, [tf.shape(convVA_sig)[0], fcVA_node])
                log_var = self.fully_connect(convVA_re, fcVA_node_scalar, self.CODE_DIMENTION, self.SEED)
                # self.var = tf.exp(var)
            return mean, log_var

        
    def unet(self, input_data, mean, log_var, reuse=False):
        with tf.variable_scope("UNet", reuse=reuse):
            with tf.variable_scope("encoder_1"):  # 128x128 -> 64x64
                self.input_data = input_data
                conv1 = self.conv2d(input_data, self.INPUT_CHANNEL, self.BASE_CHANNEL, self.CONV_FILTER_SIZE1,
                                    1, self.SEED)
                conv1 = self.batch_norm(conv1)
                conv1_relu = tf.nn.relu(conv1)
                conv1_relu = tf.nn.max_pool(conv1_relu, [1,2,2,1], [1,2,2,1], padding='SAME')

            with tf.variable_scope("encoder_2"):  # 64x64 -> 32x32
                conv2 = self.conv2d(conv1_relu, self.BASE_CHANNEL, self.BASE_CHANNEL * 2, self.CONV_FILTER_SIZE2,
                                    1, self.SEED)
                conv2 = self.batch_norm(conv2)
                conv2_relu = tf.nn.relu(conv2)
                conv2_relu = tf.nn.max_pool(conv2_relu, [1,2,2,1], [1,2,2,1], padding='SAME')

            with tf.variable_scope("encoder_3"):  # 32x32 -> 16x16
                conv3 = self.conv2d(conv2_relu, self.BASE_CHANNEL * 2, self.BASE_CHANNEL * 4, self.CONV_FILTER_SIZE2,
                                    1, self.SEED)
                conv3 = self.batch_norm(conv3)
                conv3_relu = tf.nn.relu(conv3)
                conv3_relu = tf.nn.max_pool(conv3_relu, [1,2,2,1], [1,2,2,1], padding='SAME')

            with tf.variable_scope("encoder_4"):  # 16x16 -> 8x8
                conv4 = self.conv2d(conv3_relu, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 8, self.CONV_FILTER_SIZE2,
                                    1, self.SEED)
                conv4 = self.batch_norm(conv4)
                conv4_relu = tf.nn.relu(conv4)
                conv4_relu = tf.nn.max_pool(conv4_relu, [1,2,2,1], [1,2,2,1], padding='SAME')
                
            with tf.variable_scope("encoder_5"):  # 8x8 -> 4x4
                conv5 = self.conv2d(conv4_relu, self.BASE_CHANNEL * 8, self.BASE_CHANNEL * 8, self.CONV_FILTER_SIZE2,
                                    1, self.SEED)
                conv5 = self.batch_norm(conv5)
                conv5_relu = tf.nn.relu(conv5)
                conv5_relu = tf.nn.max_pool(conv5_relu, [1,2,2,1], [1,2,2,1], padding='SAME')

            with tf.variable_scope("decoder_1"):
                output_shape_de1 = tf.stack(
                    [tf.shape(conv4_relu)[0], tf.shape(conv4_relu)[1], tf.shape(conv4_relu)[2], tf.shape(conv4_relu)[3]])
                deconv_1 = self.conv2d_transpose(conv5_relu, self.BASE_CHANNEL * 8, self.BASE_CHANNEL * 8,
                                                   self.CONV_FILTER_SIZE2,
                                                   self.CONV_STRIDE2, output_shape_de1, self.SEED)
                deconv_1 = self.batch_norm(deconv_1)
                deconv_1_relu = tf.nn.relu(deconv_1)
            #
            # with tf.variable_scope("decoder_4"):
            #     output_shape_de4 = tf.stack(
            #         [tf.shape(self.conv2_relu)[0], tf.shape(self.conv2_relu)[1], tf.shape(self.conv2_relu)[2], tf.shape(self.conv2_relu)[3]])
            #     deconv_de4 = self.conv2d_transpose(deconv_de3_relu, self.BASE_CHANNEL * 4, self.BASE_CHANNEL * 2,
            #                                        self.CONV_FILTER_SIZE2,
            #                                        self.CONV_STRIDE2, output_shape_de4, self.SEED)
            #     deconv_de4 = self.batch_norm(deconv_de4)
            #     deconv_de4_relu = tf.nn.relu(deconv_de4)
            #
            # with tf.variable_scope("decoder_5"):
            #     output_shape_de5 = tf.stack(
            #         [tf.shape(self.conv1_relu)[0], tf.shape(self.conv1_relu)[1], tf.shape(self.conv1_relu)[2], tf.shape(self.conv1_relu)[3]])
            #     deconv_de5 = self.conv2d_transpose(deconv_de4_relu, self.BASE_CHANNEL * 2, self.BASE_CHANNEL,
            #                                        self.CONV_FILTER_SIZE2,
            #                                        self.CONV_STRIDE2, output_shape_de5, self.SEED)
            #     deconv_de5 = self.batch_norm(deconv_de5)
            #     deconv_de5_relu = tf.nn.relu(deconv_de5)
            with tf.variable_scope("decoder_2"):
                con_de_2 = tf.concat([deconv_1_relu, conv4_relu], axis=3)
                output_shape_de2 = tf.stack(
                    [tf.shape(conv3_relu)[0], tf.shape(conv3_relu)[1], tf.shape(conv3_relu)[2], tf.shape(conv3_relu)[3]])
                deconv_2 = self.conv2d_transpose(con_de_2, self.BASE_CHANNEL * 16, self.BASE_CHANNEL * 4,
                                                   self.CONV_FILTER_SIZE2,
                                                   self.CONV_STRIDE2, output_shape_de2, self.SEED)
                deconv_2 = self.batch_norm(deconv_2)
                deconv_2_relu = tf.nn.relu(deconv_2)

            with tf.variable_scope("decoder_3"):
                con_de_3 = tf.concat([deconv_2_relu, conv3_relu], axis=3)
                output_shape_de3 = tf.stack(
                    [tf.shape(conv2_relu)[0], tf.shape(conv2_relu)[1], tf.shape(conv2_relu)[2], tf.shape(conv2_relu)[3]])
                deconv_3 = self.conv2d_transpose(con_de_3, self.BASE_CHANNEL * 8, self.BASE_CHANNEL * 2,
                                                   self.CONV_FILTER_SIZE2,
                                                   self.CONV_STRIDE2, output_shape_de3, self.SEED)
                deconv_3 = self.batch_norm(deconv_3)
                deconv_3_relu = tf.nn.relu(deconv_3)

            with tf.variable_scope("decoder_4"):
                con_de_4 = tf.concat([deconv_3_relu, conv2_relu], axis=3)
                output_shape_de4 = tf.stack(
                    [tf.shape(conv1_relu)[0], tf.shape(conv1_relu)[1], tf.shape(conv1_relu)[2], tf.shape(conv1_relu)[3]])
                deconv_4 = self.conv2d_transpose(con_de_4, self.BASE_CHANNEL * 4, self.BASE_CHANNEL,
                                                   self.CONV_FILTER_SIZE2,
                                                   self.CONV_STRIDE2, output_shape_de4, self.SEED)
                deconv_4 = self.batch_norm(deconv_4)
                deconv_4_relu = tf.nn.relu(deconv_4)

            with tf.variable_scope("decoder_5"):
                con_de_5 = tf.concat([deconv_4_relu, conv1_relu], axis=3)
                output_shape_de5 = tf.stack(
                    [tf.shape(input_data)[0], tf.shape(input_data)[1], tf.shape(input_data)[2], tf.shape(conv1_relu)[3]])
                deconv_5 = self.conv2d_transpose(con_de_5, self.BASE_CHANNEL * 2, self.BASE_CHANNEL,
                                                   self.CONV_FILTER_SIZE1,
                                                   self.CONV_STRIDE1, output_shape_de5, self.SEED)
                deconv_5 = self.batch_norm(deconv_5)
                deconv_5_relu = tf.nn.relu(deconv_5)

            with tf.variable_scope("sample_z"):
                eps = tf.random_normal(tf.shape(log_var), dtype=tf.float32, mean=0, stddev=1.0)
                z = mean + tf.exp(log_var / 2.0) * eps
                shape_ones = tf.stack([tf.shape(input_data)[0], tf.shape(input_data)[1], tf.shape(input_data)[2],
                                       tf.shape(z)[1]])
                ones = tf.ones(shape_ones, tf.float32)
                print("z.get_shape(), ", z.get_shape())
                z_re = tf.reshape(z, [tf.shape(z)[0], tf.constant(1), tf.constant(1), tf.shape(z)[1]])
                z_chan = z_re * ones

            with tf.variable_scope("last_conv"):
                con_last = tf.concat([deconv_5_relu, z_chan], axis=3)
                conv_last = self.conv2d(deconv_5_relu, self.BASE_CHANNEL, self.CLASS_CHANNEL, self.CONV_FILTER_SIZE1,
                                    1, self.SEED)
                out_mask = tf.nn.softmax(conv_last)

            return out_mask


    def loss(self, mean_pri, log_var_pri, mean_pos, log_var_pos, pred, tar):
        var_diag_pri = tf.linalg.diag(tf.exp(log_var_pri))
        print("var_diag_pri.get_shape(), ", var_diag_pri.get_shape())
        var_diag_pos = tf.linalg.diag(tf.exp(log_var_pos))
        print("var_diag_pos.get_shape(), ", var_diag_pos.get_shape())
        var_inverse_pri = tf.linalg.inv(var_diag_pri)
        print("var_inverse_pri.get_shape(), ", var_inverse_pri.get_shape())
        vae_cost_1term = tf.log(tf.linalg.det(var_diag_pri) / (tf.linalg.det(var_diag_pos) + 1e-10) + 1e-10)
        print("vae_cost_1term.get_shape(), ", vae_cost_1term.get_shape())
        vae_cost_2term = tf.ones([tf.shape(mean_pri)[0]], dtype=tf.float32) * (-6.0)
        print("vae_cost_2term.get_shape(), ", vae_cost_2term.get_shape())
        vae_cost_3term = tf.trace(tf.matmul(var_inverse_pri, var_diag_pos))
        print("vae_cost_3term.get_shape(), ", vae_cost_3term.get_shape())
        # debug1 = tf.transpose(mean_pri - mean_pos, perm=[1,0])
        # print("debug1.get_shape(), ", debug1.get_shape())
        mean_dis1 = tf.reshape(mean_pri - mean_pos, [tf.shape(mean_pri)[0], tf.constant(1), tf.shape(mean_pri)[1]])
        # debug2 = tf.matmul(mean_dis1, var_inverse_pri)
        # print("debug2.get_shape(), ", debug2.get_shape())
        print("mean_dis1.get_shape(), ", mean_dis1.get_shape())
        mean_dis2 = tf.reshape(mean_pri - mean_pos, [tf.shape(mean_pri)[0], tf.shape(mean_pri)[1], tf.constant(1)])
        print("mean_dis2.get_shape(), ", mean_dis2.get_shape())
        vae_cost_4term = tf.reduce_sum(tf.matmul(tf.matmul(mean_dis1, var_inverse_pri), mean_dis2), axis=[1,2])
        print("vae_cost_4term.get_shape(), ", vae_cost_4term.get_shape())
        vae_cost = tf.reduce_mean(0.5 * (vae_cost_1term + vae_cost_2term + vae_cost_3term + vae_cost_4term))
        print("vae_cost.get_shape(), ", vae_cost.get_shape())
        # vae_cost = tf.reduce_mean(0.5 *(tf.square(mean) + tf.exp(log_var) - log_var - 1.0))
        print("tar.get_shape(), ", tar.get_shape())
        print("pred.get_shape(), ", pred.get_shape())
        recon_cost = - tf.reduce_mean(tf.multiply(tar, tf.log(tf.clip_by_value(pred, 1e-10, 1.0))))
        # recon_cost = tf.reduce_mean((x - x_recon)**2)
        cost = recon_cost + self.KL_BETA * vae_cost
        return cost





