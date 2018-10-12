#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image, ImageDraw
import csv

cityScape_color_chan = np.array([
            [0.0, 0.0, 0.0],#0
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [111.0, 74.0, 0.0],  #
            [81.0, 0.0, 81.0],  #
            [128.0, 64.0, 128.0],  #
            [244.0, 35.0, 232.0],  #
            [250.0, 170.0, 160.0],  #
            [230.0, 150.0, 140.0],  #
            [70.0, 70.0, 70.0],  #
            [102.0, 102.0, 156.0],  #
            [190.0, 153.0, 153.0],  #
            [180.0, 165.0, 180.0],  #
            [150.0, 100.0, 100.0],  #
            [150.0, 120.0, 90.0],  #
            [153.0, 153.0, 153.0],  #
            [153.0, 153.0, 153.0],  #
            [250.0, 170.0, 30.0],  #
            [220.0, 220.0, 0.0],  #
            [107.0, 142.0, 35.0],  #
            [152.0, 251.0, 152.0],  #
            [70.0, 130.0, 180.0],  #
            [220.0, 20.0, 60.0],  #
            [255.0, 0.0, 0.0],  #
            [0.0, 0.0, 142.0],  #
            [0.0, 0.0, 70.0],#
            [0.0, 60.0, 100.0],  #
            [0.0, 0.0, 90.0],#
            [0.0, 0.0, 110.0],#
            [0.0, 80.0, 100.0],  #
            [0.0, 0.0, 230.0],#
            [119.0, 11.0, 32.0],#
            [0.0, 0.0, 142.0]  #
            ], dtype=np.float32
            )


def unnorm_img(img_np):
    # img_np_255 = (img_np + 1.0) * 127.5
    img_np_255 = img_np * 255.
    img_np_255_mod1 = np.maximum(img_np_255, 0)
    img_np_255_mod1 = np.minimum(img_np_255_mod1, 255)
    img_np_uint8 = img_np_255_mod1.astype(np.uint8)
    return img_np_uint8


def convert_np2pil(images_255):
    list_images_PIL = []
    for num, images_255_1 in enumerate(images_255):
        # img_255_tile = np.tile(images_255_1, (1, 1, 3))
        image_1_PIL = Image.fromarray(images_255_1)
        list_images_PIL.append(image_1_PIL)
    return list_images_PIL

def convert_seg2pil(segs):
    list_images_PIL = []

    segs_argmax = np.argmax(segs, axis=3)
    for num, segs_1 in enumerate(segs_argmax):
        segs_zero = np.zeros((segs_1.shape[0], segs_1.shape[1], 3), dtype=np.float32)
        # segs_zero = np.tile(segs_zero, (1, 1, 3))
        # img_255_tile = np.tile(images_255_1, (1, 1, 3))
        for num_h, h in enumerate(segs_1):
            for num_w, value in enumerate(h):
                segs_zero[num_h, num_w] = cityScape_color_chan[int(value)]

        image_1_PIL = Image.fromarray(segs_zero.astype(np.uint8))
        list_images_PIL.append(image_1_PIL)
    return list_images_PIL


def make_output_img(img_batch, segs, pred, epoch, log_file_name, out_img_dir):#
    #img_batch, segs, output_1, EPOCH, args.log_file_name, OUT_IMG_DIR
    (data_num, img1_h, img1_w, _) = img_batch.shape

    img_batch_1_unn = unnorm_img(img_batch)
    img_pil = convert_np2pil(img_batch_1_unn)
    segs_pil = convert_seg2pil(segs)
    pred_pil = convert_seg2pil(pred)


    wide_image_np = np.ones(((img1_h + 1) * 3 - 1, (img1_w + 1) * data_num -1, 3), dtype=np.uint8) * 255
    wide_image_PIL = Image.fromarray(wide_image_np)
    for num, (ori, seg, pred1) in enumerate(zip(img_pil, segs_pil, pred_pil)):
        wide_image_PIL.paste(ori, ((img1_w + 1) * num, 0))
        wide_image_PIL.paste(seg, ((img1_w + 1) * num, (img1_h + 1)))
        wide_image_PIL.paste(pred1, ((img1_w + 1) * num, 2 * (img1_h + 1)))


    wide_image_PIL.save(out_img_dir + "/resultImage_" + log_file_name + '_' + str(epoch) + ".png")

def cal_learning_rate_with_thr(init_lr, epoch, thr, descend_number):
    lr = init_lr / (2**(epoch // descend_number))
    return max(lr, thr)

if __name__ == '__main__':
    #debug
    filename = ''