import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import PUNet as Model
# from autoencoder import AE_pool as AE

import argparse
from make_datasets import Make_datasets_CityScape as Make_datasets
import utility as util

def parser():
    parser = argparse.ArgumentParser(description='classify ketten images')
    parser.add_argument('--batch_size', '-b', type=int, default=20, help='Number of images in each mini-batch')
    parser.add_argument('--log_file_name', '-lf', type=str, default='logPUNet01', help='log file name')
    parser.add_argument('--epoch', '-e', type=int, default=31, help='epoch')
    parser.add_argument('--base_dir', '-bd', type=str, default='/media/webfarmer/FourTBData/datasets/CityScape/',
                        help='base directory name of data-sets')
    parser.add_argument('--img_dir', '-id', type=str, default='train/image/', help='directory name of images')
    parser.add_argument('--seg_dir', '-sd', type=str, default='train/mask/', help='directory name of masks')
    parser.add_argument('--img_val_dir', '-ivd', type=str, default='test/image/', help='directory name of validation images')
    parser.add_argument('--seg_val_dir', '-svd', type=str, default='test/mask/', help='directory name of validation masks')
    # parser.add_argument('--input_image_size', '-iim', type=int, default=224, help='input image size, only 256 or 128')
    parser.add_argument('--val_number', '-vn', type=int, default=6, help='where validation data in all data...0-5')
    parser.add_argument('--out_img_span', '-ois', type=int, default=100, help='time span when output image')

    return parser.parse_args()

args = parser()

BASE_CHANNEL = 32
CODE_DIM = 6
SEED = 2018
np.random.seed(SEED)
IMG_H = 128
IMG_W = 128
IMG_CHANNEL = 3
CLASS_NUM = 35
BATCH_SIZE = args.batch_size
EPOCH = args.epoch
# NUM_CLASS = 2
# RESNET_DEPTH = 50
IMG_SIZE_BE_CROP_W = 152
IMG_SIZE_BE_CROP_H = 152
LR = 0.001
VAL_NUMBER = args.val_number
VALID_EPOCH = 5
OUT_IMG_SPAN = args.out_img_span
TB_TRAIN = 'tensorboard/tensorboard_train_' + args.log_file_name
TB_TEST = 'tensorboard/tensorboard_test_' + args.log_file_name
LOG_FILE_NAME = 'log/' + args.log_file_name
LOG_LIST = []
OUT_IMG_DIR = './out_images'


try:
    os.mkdir(OUT_IMG_DIR)
except:
    pass

try:
    os.mkdir("./log")
except:
    pass

try:
    os.mkdir("./tensorboard")
except:
    pass
#base_dir, img_width, img_height, image_dir, seg_dir, image_val_dir, seg_val_dir,
                # img_width_be_crop, img_height_be_crop, crop_flag=False
datasets = Make_datasets(args.base_dir, IMG_W, IMG_H, args.img_dir, args.seg_dir, args.img_val_dir, args.seg_val_dir,
                         IMG_SIZE_BE_CROP_W, IMG_SIZE_BE_CROP_H, crop_flag=False)


x = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_CHANNEL])
t = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, CLASS_NUM])
is_training = tf.placeholder('bool', [])
lr_p = tf.placeholder('float')

# Model
model = Model(IMG_H, IMG_W, IMG_CHANNEL, CODE_DIM, BASE_CHANNEL, CLASS_NUM)
mean_pri, log_var_pri = model.priorNet(x)
mean_pos, log_var_pos = model.posteriorNet(x, t)

out_learn = model.unet(x, mean_pos, log_var_pos, reuse=False)
# out_infer = model.unet(x, mean_pri, log_var_pri, reuse=True)

with tf.variable_scope("loss"):
    # loss = model.loss(output, y)
    loss = model.loss(mean_pri, log_var_pri, mean_pos, log_var_pos, out_learn, t)

# with tf.variable_scope("argmax"):
#     out_argmax = tf.argmax(out_infer, axis=3)

with tf.variable_scope("train"):
    # train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    train_op = tf.train.AdamOptimizer(lr_p).minimize(loss)

# Summaries
tf.summary.scalar('train_loss', loss)
merged_summary = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(TB_TRAIN)
val_writer = tf.summary.FileWriter(TB_TEST)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer.add_graph(sess.graph)

    # model.load_original_weights(sess, skip_layers=train_layers)

    LOG_LIST.append(['epoch', 'Loss'])

    #before learning
    for epoch in range(EPOCH):
        lr_now = util.cal_learning_rate_with_thr(0.0001, epoch, 0.00005, 100)
        sum_loss = np.float32(0)

        len_data = datasets.make_data_for_1_epoch()

        for i in range(0, len_data, BATCH_SIZE):
            print("i, ", i)
            img_batch, seg_batch = datasets.get_data_for_1_batch(i, BATCH_SIZE)
            #if  i == 0...not learn
            if epoch != 0:
                sess.run(train_op, feed_dict={x: img_batch, t: seg_batch, is_training: True, lr_p:lr_now})

            s = sess.run(merged_summary, feed_dict={x: img_batch, t: seg_batch, is_training: False})
            train_writer.add_summary(s, epoch)

            loss_ = sess.run(loss, feed_dict={x: img_batch, t: seg_batch, is_training: False})
            sum_loss += loss_ * len(img_batch)

        print("----------------------------------------------------------------------")
        print("epoch = {:}, Training Loss = {:.4f}".format(epoch, sum_loss / len_data))

        if epoch % OUT_IMG_SPAN == 0:
            img_batch, segs = datasets.get_data_for_1_batch_val(0, 6)
            # img_batch2 = datasets.get_data_for_1_batch_val(0, 6)
            # print("img_batch.shape, ", img_batch.shape)

            output_1 = sess.run(out_learn, feed_dict={x: img_batch, is_training: False})
            # output_2 = sess.run(output, feed_dict={x: img_batch2, is_training: False})

            util.make_output_img(img_batch, segs, output_1, EPOCH, args.log_file_name, OUT_IMG_DIR)

    '''
    # after learning process
    fc3_list = []
    len_data = datasets.make_data_for_1_epoch()
    for i in range(0, 6, 6):
        img_batch1 = datasets.get_data_for_1_batch_val(i, BATCH_SIZE, 1)
        img_batch2 = datasets.get_data_for_1_batch_val(i, BATCH_SIZE, 2)
        print("img_batch1.shape, ", img_batch1.shape)

        output_1 = sess.run(output, feed_dict={x: img_batch1, is_training: False})
        output_2 = sess.run(output, feed_dict={x: img_batch2, is_training: False})

        util.make_output_img(img_batch1, img_batch2, output_1, output_2, EPOCH, args.log_file_name, OUT_IMG_DIR)
    '''











