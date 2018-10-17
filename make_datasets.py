import numpy as np
import os
import random
from PIL import Image
import cv2

class Make_datasets_CityScape():
    def __init__(self, base_dir, img_width, img_height, image_dir, seg_dir, image_val_dir, seg_val_dir,
                 img_width_be_crop, img_height_be_crop, crop_flag=True):
        # self.cityScapeData_H = 1024
        # self.cityScapeData_W = 2048
        self.cityScapeData_H = 256
        self.cityScapeData_W = 512
        self.base_dir = base_dir
        self.img_width = img_width
        self.img_height = img_height
        self.img_width_be_crop = img_width_be_crop
        self.img_height_be_crop = img_height_be_crop
        self.dir_img = base_dir + image_dir
        self.dir_seg = base_dir + seg_dir
        self.dir_val_img = base_dir + image_val_dir
        self.dir_val_seg = base_dir + seg_val_dir
        self.crop_flag = crop_flag
        # self.file_listX = os.listdir(self.dirX)
        # self.file_listY = os.listdir(self.dirY)
        self.file_list = self.get_file_names(self.dir_img)
        self.file_list.sort()
        self.file_val_list = self.get_file_names(self.dir_val_img)
        self.file_val_list.sort()
        self.cityScape_color_chan = np.array([
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
        print("self.cityScape_color_chan[3]", self.cityScape_color_chan[3])

        self.image_file_num = len(self.file_list)
        self.image_val_file_num = len(self.file_val_list)
        print("self.img_width", self.img_width)
        print("self.img_height", self.img_height)
        print("len(self.file_list)", len(self.file_list))
        print("self.image_file_num", self.image_file_num)
        print("self.image_val_file_num, ", self.image_val_file_num)
        print("self.base_dir = ", self.base_dir)
        print("self.dir_img, ", self.dir_img)
        print("self.dir_seg, ", self.dir_seg)
        print("self.crop.flag, ", self.crop_flag)

    def get_file_names(self, dir_name):
        target_files = []
        for root, dirs, files in os.walk(dir_name):
            targets = [os.path.join(root, f) for f in files]
            target_files.extend(targets)

        return target_files


    def get_only_img(self, list, extension):
        list_mod = []
        for y in list:
            if (y[-9:] == extension): #only .png
                list_mod.append(y)
        return list_mod


    def read_1_data(self, dir, filename_list, width, height, width_be_crop, height_be_crop, margin_H_batch,
                    margin_W_batch, crop_flag, seg_flag=False):
        images = []
        for num, filename in enumerate(filename_list):
            if seg_flag:
                _ , dir_name, file_name_only = filename.rsplit("/", 2)
                str_base, _ = file_name_only.rsplit("_",1)
                filename_seg = str_base + "_gtFine_oneHotQua.npy"
                npy_ori = np.load(dir + dir_name + '/' + filename_seg)
                # print("np.max(npy_ori), ", np.max(npy_ori))
                # print("np.min(npy_ori), ", np.min(npy_ori))

                # print("npy_ori.shape,", npy_ori.shape)
                # print("npy_ori.dtype, ", npy_ori.dtype)
                # #debug
                # npy_ori_debug = npy_ori[:,:,:3].astype(np.uint8)
                # print("npy_ori_debug.shape, ", npy_ori_debug.shape)
                # npy_resize_debug = cv2.resize(npy_ori_debug, (width_be_crop, height_be_crop))
                # print("npy_resize_debug.shape, ", npy_resize_debug.shape)
            else:

                pilIn = Image.open(filename)

            if seg_flag:
                if crop_flag:
                    # downsample_H = self.cityScapeData_H // height_be_crop
                    # downsample_W = self.cityScapeData_W // width_be_crop
                    # npy_resize = npy_ori[::downsample_H, ::downsample_W]
                    npy_resize = cv2.resize(npy_ori.astype(np.uint8), (width_be_crop, height_be_crop))
                    npy_Resize = npy_resize[margin_H_batch[num]:height + margin_H_batch[num], margin_W_batch[num]:width + margin_W_batch[num]]

                else:
                    # downsample_H = self.cityScapeData_H // height
                    # downsample_W = self.cityScapeData_W // width
                    # npy_Resize = npy_ori[::downsample_H, ::downsample_W]
                    npy_Resize = cv2.resize(npy_ori.astype(np.uint8), (width, height))
                    # pilResize = pilIn.resize((width, height))
                image = npy_Resize.astype(np.float32)
                # image = image[:,:,:3]
                # image = self.convert_color_to_indexInt(image)
                # image_t = image
            else:
                if crop_flag:
                    pilIn = pilIn.resize((width_be_crop, height_be_crop))
                    pilResize = self.crop_img(pilIn, width, height, margin_W_batch[num], margin_H_batch[num])
                    # print("pilResize.size", pilResize.size)
                else:
                    pilResize = pilIn.resize((width, height))
                image = np.asarray(pilResize, dtype=np.float32)
            #     image_t = np.transpose(image, (2, 0, 1))
            # # except:
            # #     print("filename =", filename)
            # #     image_t = image.reshape(image.shape[0], image.shape[1], 1)
            # #     image_t = np.tile(image_t, (1, 1, 3))
            # #     image_t = np.transpose(image_t, (2, 0, 1))
            images.append(image)
        # print("images.shape", np.asarray(images).shape)
        return np.asarray(images)


    def normalize_data(self, data):
        # data0_2 = data / 127.5
        # data_norm = data0_2 - 1.0
        data_norm = data / 255.0
        # data_norm = data - 1.0
        return data_norm


    def crop_img(self, data, output_img_W, output_img_H, margin_W, margin_H):
        cropped_img = data.crop((margin_W, margin_H, margin_W + output_img_W, margin_H + output_img_H))
        return cropped_img


    def read_1_data_and_convert_RGB(self, dir, filename_list, extension, width, height):
        images = []
        for filename in filename_list:
            pilIn = Image.open(dir + filename[0] + extension).convert('RGB')
            pilResize = pilIn.resize((width, height))
            image = np.asarray(pilResize)
            image_t = np.transpose(image, (2, 0, 1))
            images.append(image_t)
        return np.asarray(images)


    def write_data_to_img(self, dir, np_arrays, extension):

        for num, np_array in enumerate(np_arrays):
            pil_img = Image.fromarray(np_array)
            pil_img.save(dir + 'debug_' + str(num) + extension)


    def convert_indexInt_to_color_for_oneHot(self, data):
        # print("data.shape", data.shape)
        # print("data[0][0]", data[0][0])
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.int32)
        for h, row in enumerate(data):
            for w, col in enumerate(row):
                for c, ele in enumerate(col):
                    if ele == 1.0:

                        d_mod[h][w] = self.cityScape_color_chan[c]

        return d_mod


    def make_data_for_1_epoch(self):
        self.image_files_1_epoch = random.sample(self.file_list, self.image_file_num)
        self.margin_H = np.random.randint(0, (self.img_height_be_crop - self.img_height + 1), self.image_file_num)
        self.margin_W = np.random.randint(0, (self.img_width_be_crop - self.img_width + 1), self.image_file_num)
        # print("self.margin_H", self.margin_H)
        # print("self.margin_W", self.margin_W)
        return len(self.image_files_1_epoch)

    def get_data_for_1_batch(self, i, batchsize, train_FLAG=True):

        data_batch = self.image_files_1_epoch[i:i + batchsize]

        margin_H_batch = self.margin_H[i:i + batchsize]
        margin_W_batch = self.margin_W[i:i + batchsize]

        images = self.read_1_data(self.dir_img, data_batch, self.img_width, self.img_height, self.img_width_be_crop,
                                   self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=False)
        # imagesY = self.read_1_data(self.dirY, data_batchY, self.img_width, self.img_height, self.img_width_be_crop,
        #                            self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=False)
        segs = self.read_1_data(self.dir_seg, data_batch, self.img_width, self.img_height,
                self.img_width_be_crop, self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=True)
        #print("segs.shape", segs.shape)
        # # imagesY_seg = self.read_1_data(self.dirY_seg, data_batchY, self.img_width, self.img_height,
        # #         self.img_width_be_crop, self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=True)
        images_n = self.normalize_data(images)
        # imagesY_n = self.normalize_data(imagesY)

        # imagesX_n_seg = self.normalize_data(imagesX_seg)
        # imagesY_n_seg = self.normalize_data(imagesY_seg)

        return images_n, segs


    def get_data_for_1_batch_val(self, i, batchsize):

        data_batch = self.file_val_list[i:i + batchsize]
        margin_H_batch = self.margin_H[i:i + batchsize]
        margin_W_batch = self.margin_W[i:i + batchsize]

        images = self.read_1_data(self.dir_val_img, data_batch, self.img_width, self.img_height, self.img_width_be_crop,
                                   self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=False)
        # imagesY = self.read_1_data(self.dirY, data_batchY, self.img_width, self.img_height, self.img_width_be_crop,
        #                            self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=False)
        segs = self.read_1_data(self.dir_val_seg, data_batch, self.img_width, self.img_height,
                self.img_width_be_crop, self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=True)
        #print("segs.shape", segs.shape)
        # # imagesY_seg = self.read_1_data(self.dirY_seg, data_batchY, self.img_width, self.img_height,
        # #         self.img_width_be_crop, self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=True)
        images_n = self.normalize_data(images)
        # imagesY_n = self.normalize_data(imagesY)

        # imagesX_n_seg = self.normalize_data(imagesX_seg)
        # imagesY_n_seg = self.normalize_data(imagesY_seg)

        return images_n, segs


    def make_img_from_label(self, labels, epoch):#labels=(first_number, last_number + 1)
        labels_train = self.train_files_1_epoch[labels[0]:labels[1]]
        labels_val = self.list_val_files[labels[0]:labels[1]]
        labels_train_val = labels_train + labels_val
        labels_img_np = self.read_1_data_and_convert_RGB(self.SegmentationClass_dir, labels_train_val, '.png', self.img_width, self.img_height)
        self.write_data_to_img('debug/label_' + str(epoch) + '_',  labels_img_np, '.png')


    def make_img_from_prob(self, probs, epoch):#probs=(data, height, width)..0-20 value
        # print("probs[0]", probs[0])
        print("probs[0].shape", probs[0].shape)
        probs_RGB = util.convert_indexColor_to_RGB(probs)
        # labels_img_np = self.read_1_data_and_convert_RGB(self.SegmentationClass_dir, probs_RGB, '.jpg', self.img_width, self.img_height)
        self.write_data_to_img('debug/prob_' + str(epoch), probs_RGB, '.jpg')


    def get_concat_img_h(self, img1, img2):
        dst = Image.new('RGB', (img1.width + img2.width, img1.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (img1.width, 0))
        return dst


    def get_concat_img_w(self, img1, img2):
        dst = Image.new('RGB', (img1.width, img1.height + img2.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (0, img1.height))
        return dst


    def make_img_from_seg_prob(self, img_X, probsX, segsX, img_Y, probsY, segsY, out_image_dir, epoch, LOG_FILE_NAME):
        # print("in make_img_from_seg_prob()")
        # print("img_X.shape, ", img_X.shape)
        # # print("img_Y.shape, ", img_Y.shape)
        # print("probsX.shape, ", probsX.shape)
        # # print("probsY.shape, ", probsY.shape)
        # print("segsX.shape, ", segsX.shape)
        # print("segsY.shape, ", segsY.shape)

        # # probs_transX = np.transpose(probsX, (0, 2, 3, 1))
        # probs_argmaxX = np.argmax(probsX, axis=3)
        # print("probs_argmaxX.shape", probs_argmaxX.shape)
        # # probs_transY = np.transpose(probsY, (0, 2, 3, 1))
        # probs_argmaxY = np.argmax(probsY, axis=3)
        # print("probs_argmaxY.shape", probs_argmaxY.shape)
        # segs_transX = np.transpose(segsX, (0, 2, 3, 1))
        # segs_argmaxX = np.argmax(segsX, axis=3)
        # print("segs_argmaxX.shape", segs_argmaxX.shape)
        # segs_argmaxY = np.argmax(segsY, axis=3)
        # print("segs_argmaxY.shape", segs_argmaxY.shape)

        probs_segX = []
        for num, prob in enumerate(probsX):
            probs_segX.append(self.convert_indexInt_to_color(prob))
        probs_segX_np = np.array(probs_segX, dtype=np.float32)

        probs_segY = []
        for num, prob in enumerate(probsY):
            probs_segY.append(self.convert_indexInt_to_color(prob))
        probs_segY_np = np.array(probs_segY, dtype=np.float32)

        # probs_segY = []
        # for num, prob in enumerate(probsY):
        #     probs_segY.append(self.convert_indexInt_to_color(prob))
        # probs_segY_np = np.array(probs_segY, dtype=np.float32)

        segX = []
        for num, prob in enumerate(segsX):
            segX.append(self.convert_indexInt_to_color_for_oneHot(prob))
        segX_np = np.array(segX, dtype=np.float32)

        segY = []
        for num, prob in enumerate(segsY):
            segY.append(self.convert_indexInt_to_color_for_oneHot(prob))
        segY_np = np.array(segY, dtype=np.float32)

        # segY = []
        # for num, prob in enumerate(segsY):
        #     segY.append(self.convert_indexInt_to_color(prob))
        # segY_np = np.array(segY, dtype=np.float32)


        #img_X, img_X2Y, img_X2Y2X, img_Y, img_Y2X, img_Y2X2Y, out_image_dir, epoch, log_file_name
        wide_image1 = util.make_output_img_seg(img_X, probs_segX_np, segX_np, out_image_dir, epoch, LOG_FILE_NAME)
        wide_image2 = util.make_output_img_seg(img_Y, probs_segY_np, segY_np, out_image_dir, epoch, LOG_FILE_NAME)
        util.make_output_img_and_save(wide_image1, wide_image2, out_image_dir, epoch, LOG_FILE_NAME)


    def make_2chanAnno_from_3chanImg(np_array):
        np_array_1chan = np_array[:, :, 0].reshape(np_array.shape[0], np_array.shape[1], 1).astype(np.float32)

        np_array_1chan_re = 1.0 - np_array_1chan

        np_array_2canAnno = np.concatenate((np_array_1chan, np_array_1chan_re), axis=2)
        return np_array_2canAnno


class Make_datasets_MNIST():

    def __init__(self, file_name, img_width, img_height, seed):
        self.filename = file_name
        self.img_width = img_width
        self.img_height = img_height
        self.seed = seed
        x_train, x_test, x_valid, y_train, y_test, y_valid = self.read_MNIST_npy(self.filename)
        self.train_np = np.concatenate((y_train.reshape(-1,1), x_train), axis=1).astype(np.float32)
        self.test_np = np.concatenate((y_test.reshape(-1,1), x_test), axis=1).astype(np.float32)
        self.valid_np = np.concatenate((y_valid.reshape(-1,1), x_valid), axis=1).astype(np.float32)
        print("self.train_np.shape, ", self.train_np.shape)
        print("self.test_np.shape, ", self.test_np.shape)
        print("self.valid_np.shape, ", self.valid_np.shape)
        print("np.max(x_train), ", np.max(x_train))
        print("np.min(x_train), ", np.min(x_train))
        self.train_data_5, self.train_data_7 = self.divide_MNIST_by_digit(self.train_np, 5, 7)
        print("self.train_data_5.shape, ", self.train_data_5.shape)
        print("self.train_data_7.shape, ", self.train_data_7.shape)
        self.valid_data_5, self.valid_data_7 = self.divide_MNIST_by_digit(self.valid_np, 5, 7)
        print("self.valid_data_5.shape, ", self.valid_data_5.shape)
        print("self.valid_data_7.shape, ", self.valid_data_7.shape)
        self.valid_data_5_7 = np.concatenate((self.train_data_7, self.valid_data_7, self.valid_data_5))

        random.seed(self.seed)
        np.random.seed(self.seed)


    def read_MNIST_npy(self, filename):
        mnist_npz = np.load(filename)
        print("type(mnist_npz), ", type(mnist_npz))
        print("mnist_npz.keys(), ", mnist_npz.keys())
        print("mnist_npz['x_train'].shape, ", mnist_npz['x_train'].shape)
        print("mnist_npz['x_test'].shape, ", mnist_npz['x_test'].shape)
        print("mnist_npz['x_valid'].shape, ", mnist_npz['x_valid'].shape)
        print("mnist_npz['y_train'].shape, ", mnist_npz['y_train'].shape)
        print("mnist_npz['y_test'].shape, ", mnist_npz['y_test'].shape)
        print("mnist_npz['y_valid'].shape, ", mnist_npz['y_valid'].shape)
        x_train = mnist_npz['x_train']
        x_test = mnist_npz['x_test']
        x_valid = mnist_npz['x_valid']
        y_train = mnist_npz['y_train']
        y_test = mnist_npz['y_test']
        y_valid = mnist_npz['y_valid']
        return x_train, x_test, x_valid, y_train, y_test, y_valid


    def get_file_names(self, dir_name):
        target_files = []
        for root, dirs, files in os.walk(dir_name):
            targets = [os.path.join(root, f) for f in files]
            target_files.extend(targets)

        return target_files

    def divide_MNIST_by_digit(self, train_np, data1_num, data2_num):
        data_1 = train_np[train_np[:,0] == data1_num]
        data_2 = train_np[train_np[:,0] == data2_num]

        return data_1, data_2



    def read_data(self, d_y_np, width, height):
        tars = []
        images = []
        for num, d_y_1 in enumerate(d_y_np):
            image = d_y_1[1:].reshape(width, height, 1)
            tar = d_y_1[0]
            images.append(image)
            tars.append(tar)

        return np.asarray(images), np.asarray(tars)


    def normalize_data(self, data):
        # data0_2 = data / 127.5
        # data_norm = data0_2 - 1.0
        data_norm = (data * 2.0) - 1.0 #applied for tanh

        return data_norm


    def make_data_for_1_epoch(self):
        self.filename_1_epoch = np.random.permutation(self.train_data_5)

        return len(self.filename_1_epoch)


    def get_data_for_1_batch(self, i, batchsize):
        filename_batch = self.filename_1_epoch[i:i + batchsize]
        images, _ = self.read_data(filename_batch, self.img_width, self.img_height)
        images_n = self.normalize_data(images)
        return images_n

    def get_valid_data_for_1_batch(self, i, batchsize):
        filename_batch = self.valid_data_5_7[i:i + batchsize]
        images, tars = self.read_data(filename_batch, self.img_width, self.img_height)
        images_n = self.normalize_data(images)
        return images_n, tars

    def make_random_z_with_norm(self, mean, stddev, data_num, unit_num):
        norms = np.random.normal(mean, stddev, (data_num, unit_num))
        # tars = np.zeros((data_num, 1), dtype=np.float32)
        return norms


    def make_target_1_0(self, value, data_num):
        if value == 0.0:
            target = np.zeros((data_num, 1), dtype=np.float32)
        elif value == 1.0:
            target = np.ones((data_num, 1), dtype=np.float32)
        else:
            print("target value error")
        return target


class Make_datasets_AE():
    def __init__(self, base_dir, img_dir1, img_dir2, img_width, img_height, img_width_be_crop, img_height_be_crop,
                 crop_flag=False, val_num=6, flip_flag=False, predict_flag=False, predict_img=None):

        self.base_dir = base_dir
        self.img_dir1 = self.base_dir + img_dir1
        self.img_dir2 = self.base_dir + img_dir2
        self.img_width = img_width
        self.img_height = img_height
        self.img_width_be_crop = img_width_be_crop
        self.img_height_be_crop = img_height_be_crop
        self.val_num = val_num
        self.predict_flag = predict_flag
        self.predict_img = predict_img
        self.crop_flag = crop_flag
        self.flip_flag = flip_flag
        self.file_list1 = self.get_file_names(self.img_dir1)
        self.file_list1.sort()
        self.file_list2 = self.get_file_names(self.img_dir2)
        self.file_list2.sort()
        self.image1_file_num = len(self.file_list1)
        self.image2_file_num = len(self.file_list2)
        print("self.base_dir = ", self.base_dir)
        print("self.img_dir2 = ", self.img_dir2)
        print("self.img_width", self.img_width)
        print("self.img_height", self.img_height)
        print("len(self.file_list1)", len(self.file_list1))
        print("len(self.file_list2)", len(self.file_list2))
        print("self.image1_file_num", self.image1_file_num)
        print("self.image2_file_num", self.image2_file_num)

        self.train_file_list = self.file_list1[self.val_num:]
        self.img1_val_file_list = self.file_list1[:self.val_num]
        self.train_file_num = len(self.train_file_list)
        self.img1_val_file_num = len(self.img1_val_file_list)
        print("self.train_file_num, ", self.train_file_num)
        print("self.img1_val_file_num, ", self.img1_val_file_num)

    def get_file_names(self, dir_name):
        target_files = []
        for root, dirs, files in os.walk(dir_name):
            targets = [os.path.join(root, f) for f in files]
            target_files.extend(targets)

        return target_files

    def get_only_img(self, list, extension):
        list_mod = []
        for y in list:
            if (y[-9:] == extension):  # only .png
                list_mod.append(y)
        return list_mod

    def get_only_img_png(self, list):
        list_mod = []
        for y in list:
            dir_name, file_name_only = y.rsplit("/", 1)
            if (file_name_only == 'img.png'):  # only .png
                list_mod.append(y)
        return list_mod

    def read_1_data(self, filename_list, width, height, width_be_crop, height_be_crop, margin_H_batch,
                    margin_W_batch, crop_flag, flip_flag=False, flip_list=None):
        images = []
        for num, filename in enumerate(filename_list):
            pilIn = Image.open(filename).convert("RGB")
            self.ori_img_width, self.ori_img_height = pilIn.size

            if crop_flag:
                pilIn = pilIn.resize((width_be_crop, height_be_crop))
                pilResize = self.crop_img(pilIn, width, height, margin_W_batch[num], margin_H_batch[num])
            else:
                pilResize = pilIn.resize((width, height))

            image = np.asarray(pilResize, dtype=np.float32)

            if flip_flag:
                image = self.flip_image(image, flip_list[num])

            images.append(image)
        return np.asarray(images)


    def flip_image(self, image, flip_value):  # flip_value is expected 0 or 1 or 2 or 3
        image_np = image
        if flip_value % 2 == 1:  # 1, 3....flip vertically
            image_np = image_np[::-1, :, :]
        if flip_value // 2 == 1:  # 2, 3...flip horizontally
            image_np = image_np[:, ::-1, :]
        return image_np

    def normalize_data(self, data):
        data0_2 = data / 127.5
        data_norm = data0_2 - 1.0
        # data_norm = data / 255.0
        # data_norm = data - 1.0
        return data_norm

    def crop_img(self, data, output_img_W, output_img_H, margin_W, margin_H):
        cropped_img = data.crop((margin_W, margin_H, margin_W + output_img_W, margin_H + output_img_H))
        return cropped_img

    def make_data_for_1_epoch(self):
        self.image_files_1_epoch = random.sample(self.train_file_list, self.train_file_num)
        self.margin_H = np.random.randint(0, (self.img_height_be_crop - self.img_height + 1), self.train_file_num)
        self.margin_W = np.random.randint(0, (self.img_width_be_crop - self.img_width + 1), self.train_file_num)
        self.flip_list = np.random.randint(0, 4, self.train_file_num)
        # print("self.margin_H", self.margin_H)
        # print("self.margin_W", self.margin_W)
        return len(self.image_files_1_epoch)

    def get_data_for_1_batch(self, i, batchsize, train_FLAG=True):

        data_batch = self.image_files_1_epoch[i:i + batchsize]

        margin_H_batch = self.margin_H[i:i + batchsize]
        margin_W_batch = self.margin_W[i:i + batchsize]
        flip_list_batch = self.flip_list[i:i + batchsize]

        images = self.read_1_data(data_batch, self.img_width, self.img_height, self.img_width_be_crop,
                                  self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag,
                                  self.flip_flag, flip_list_batch)

        images_n = self.normalize_data(images)

        return images_n

    def get_data_for_1_batch_val(self, i, batchsize, img1_or_2):
        if img1_or_2 == 1:
            data_batch = self.img1_val_file_list[i:i + batchsize]
        else:
            data_batch = self.file_list2[i:i + batchsize]

        margin_H_batch = self.margin_H[i:i + batchsize]
        margin_W_batch = self.margin_W[i:i + batchsize]
        flip_list_batch = np.zeros(batchsize, dtype=np.int32)

        images = self.read_1_data(data_batch, self.img_width, self.img_height, self.img_width_be_crop,
                                  self.img_height_be_crop, margin_H_batch, margin_W_batch, False,
                                  self.flip_flag, flip_list_batch)

        images_n = self.normalize_data(images)

        return images_n


    def make_random_z_with_norm(self, mean, stddev, data_num, unit_num):
        norms = np.random.normal(mean, stddev, (data_num, unit_num))
        # tars = np.zeros((data_num, 1), dtype=np.float32)
        return norms


    def make_target_1_0(self, value, data_num):
        if value == 0.0:
            target = np.zeros((data_num, 1), dtype=np.float32)
        elif value == 1.0:
            target = np.ones((data_num, 1), dtype=np.float32)
        else:
            print("target value error")
        return target


class Make_datasets_WallCrack_labelme():
    def __init__(self, base_dir, img_width, img_height, image_dir, seg_dir, image_val_dir, seg_val_dir, img_width_be_crop,
                 img_height_be_crop,
                 crop_flag=True, val_num=0, flip_flag=True, predict_flag=False, predict_img=None, rotate_flag=False,
                 mixup_flag=False, mixup_rate=1.0, mixup_alpha=0.4, random_erasing_flag=True):

        self.base_dir = base_dir  # /media/webfarmer/HDCZ-UT/dataset/wall/training/
        # self.test_dir = test_dir  # /media/webfarmer/HDCZ-UT/dataset/wall/test/
        self.img_width = img_width
        self.img_height = img_height
        self.img_width_be_crop = img_width_be_crop
        self.img_height_be_crop = img_height_be_crop
        self.dir_img = base_dir + image_dir  # /media/webfarmer/HDCZ-UT/dataset/wall/training/original_data/
        self.dir_seg = base_dir + seg_dir  # /media/webfarmer/HDCZ-UT/dataset/wall/trainin/annotation/
        self.dir_test_img = base_dir + image_val_dir  # /media/webfarmer/HDCZ-UT/dataset/wall/test/original_data/
        self.dir_test_seg = base_dir + seg_val_dir  # /media/webfarmer/HDCZ-UT/dataset/wall/test/annotation/
        self.val_num = val_num
        self.predict_flag = predict_flag
        self.predict_img = predict_img
        self.rotate_flag = rotate_flag
        self.crop_flag = crop_flag
        self.flip_flag = flip_flag
        self.mixup_flag = mixup_flag
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha
        self.random_erasing_flag = random_erasing_flag
        file_list_be = self.get_file_names(self.dir_img)
        self.file_list = self.get_only_img_png(file_list_be)
        self.file_list.sort()
        # debug
        self.print_file_list(self.file_list)
        self.file_test_list = self.get_file_names(self.dir_test_img)
        self.file_test_list = self.get_only_img_png(self.file_test_list)
        self.file_test_list.sort()
        self.wallCrack_color_chan = np.array([
            [255.0, 0.0, 0.0],  # class 0
            [0.0, 0.0, 0.0],  # class 1
            [0.0, 0.0, 0.0],  # class 2
            [0.0, 0.0, 0.0],  # class 3
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]], dtype=np.float32)

        self.image_file_num = len(self.file_list)
        self.image_test_file_num = len(self.file_test_list)
        print("self.base_dir = ", self.base_dir)
        # print("self.test_dir = ", self.test_dir)
        print("self.dir_img, ", self.dir_img)
        print("self.dir_seg, ", self.dir_seg)
        print("self.dir_test_img, ", self.dir_test_img)
        print("self.dir_test_seg, ", self.dir_test_seg)
        print("self.img_width", self.img_width)
        print("self.img_height", self.img_height)
        print("len(self.file_list)", len(self.file_list))
        print("len(self.file_test_list)", len(self.file_test_list))
        print("self.image_file_num", self.image_file_num)
        print("self.image_test_file_num", self.image_test_file_num)
        self.train_file_list = self.file_list[self.val_num:]
        self.val_file_list = self.file_list[:self.val_num]
        # self.train_file_list = self.file_list[:len(self.file_list) - self.val_num]
        # self.val_file_list = self.file_list[len(self.file_list) - self.val_num:]
        self.train_file_num = len(self.train_file_list)
        self.val_file_num = len(self.val_file_list)
        print("self.train_file_num, ", self.train_file_num)
        print("self.val_file_num, ", self.val_file_num)
        print("self.mixup_flag, ", self.mixup_flag)

    def get_file_names(self, dir_name):
        target_files = []
        for root, dirs, files in os.walk(dir_name):
            targets = [os.path.join(root, f) for f in files]
            target_files.extend(targets)

        return target_files

    def get_only_img(self, list, extension):
        list_mod = []
        for y in list:
            if (y[-9:] == extension):  # only .png
                list_mod.append(y)
        return list_mod

    def get_only_img_png(self, list):
        list_mod = []
        for y in list:
            dir_name, file_name_only = y.rsplit("/", 1)
            if (file_name_only == 'img.png'):  # only .png
                list_mod.append(y)
        return list_mod

    def print_file_list(self, list):
        for num, filename in enumerate(list):
            dir_name, indi_dir, file_name_only = filename.rsplit("/", 2)
            print(indi_dir)

    def read_data_to_np(self, dir, filename_list, width, height, width_be_crop, height_be_crop, margin_H_batch,
                        margin_W_batch, crop_flag, flip_flag=False, flip_list=None, rotate_flag=False, rotate_list=None,
                        mixup_flag=False, random_erasing_flag=False):
        images = []
        tars = []
        for num, filename in enumerate(filename_list):
            dir_name, file_name_only = filename.rsplit("/", 1)  # ex) ...annotation01_180812, img.png
            tarImg = Image.open(dir_name + '/' + 'label.png')
            oriImg = Image.open(filename).convert("RGB")
            self.ori_img_width, self.ori_img_height = oriImg.size

            if crop_flag:
                before_crop_tar = tarImg.resize((width_be_crop, height_be_crop))
                tar_Resize = self.crop_img(before_crop_tar, width, height, margin_W_batch[num], margin_H_batch[num])
                before_crop_Img = oriImg.resize((width_be_crop, height_be_crop))
                ori_Resize = self.crop_img(before_crop_Img, width, height, margin_W_batch[num], margin_H_batch[num])
            else:
                tar_Resize = tarImg.resize((width, height))
                ori_Resize = oriImg.resize((width, height))
            tar_be = np.asarray(tar_Resize, dtype=np.float32)
            # image = self.make_4chanAnno_from_3chanImg(image_be)
            tar = self.make_2chanAnno_from_3chanImg(tar_be)
            image = np.asarray(ori_Resize, dtype=np.float32)

            if flip_flag:
                tar = self.flip_image(tar, flip_list[num])
                image = self.flip_image(image, flip_list[num])

            if rotate_flag:
                tar = self.rotate_image(tar, rotate_list[num])
                image = self.rotate_image(image, rotate_list[num])

            images.append(image)
            tars.append(tar)
        images = np.asarray(images, dtype=np.float32)
        tars = np.asarray(tars, dtype=np.float32)

        if mixup_flag:
            images, tars = self.do_mixup(images, tars, self.mixup_rate, self.mixup_alpha)

        if random_erasing_flag:
            images, tars = self.do_random_erasing(images, tars)

        return images, tars

    def do_mixup(self, data, tar, mixup_rate, alpha):  # mixup_rate is expected (0, 1)
        if len(data) < 2:
            return data, tar
        mixup_num = int(len(data) * mixup_rate / 2)
        if mixup_num == 0:
            return data, tar
        data_mixup1 = data[:mixup_num]
        data_mixup2 = data[mixup_num:(mixup_num * 2)]
        data_not_mixup = data[(mixup_num * 2):]
        tar_mixup1 = tar[:mixup_num]
        tar_mixup2 = tar[mixup_num:(mixup_num * 2)]
        tar_not_mixup = tar[(mixup_num * 2):]
        self.lam = self.beta_func(alpha, mixup_num).reshape(-1, 1, 1, 1)
        # print("self.lam, ", self.lam)
        mixuped_data = self.lam * data_mixup1 + (1 - self.lam) * data_mixup2  # do mixup
        mixuped_tar = self.lam * tar_mixup1 + (1 - self.lam) * tar_mixup2  # do mixup
        data_con = np.concatenate((mixuped_data, data_not_mixup), axis=0)
        tar_con = np.concatenate((mixuped_tar, tar_not_mixup), axis=0)
        return data_con, tar_con

    def do_random_erasing(self, data, tar, prob=1.0, sl=0.02, sh=0.4, r1=0.3):  # random erasing implementation
        do_R_E_num = int(len(data) * prob)
        random_per = np.random.randint(0, len(data), len(data))
        data_per = data[random_per]
        tar_per = tar[random_per]

        # get variants
        re_0_1 = np.random.rand(do_R_E_num)
        # print("re_0_1, ", re_0_1)
        r2 = 1.0
        re = re_0_1 * (r2 - r1) + r1
        # print("re, ", re)
        se_0_1 = np.random.rand(do_R_E_num)
        se = (data.shape[1] * data.shape[2]) * (sl + se_0_1 * (sh - sl))
        he = ((se * re) ** 0.5).astype(np.int32)
        we = ((se / re) ** 0.5).astype(np.int32)
        # print("he, ", he)
        # print("we, ", we)
        margin_h = data.shape[1] - he
        margin_w = data.shape[2] - we
        # print("margin_h, ", margin_h)
        # print("margin_w, ", margin_w)
        margin_rand = np.random.rand(do_R_E_num, 2)

        xe = (margin_w * margin_rand[:, 0]).astype(np.int32)
        ye = (margin_h * margin_rand[:, 1]).astype(np.int32)

        rand_color = np.random.randint(0, 255, do_R_E_num)

        for num, data1, in enumerate(data_per):
            if do_R_E_num <= num:
                break
            data1[ye[num]:ye[num] + he[num], xe[num]:xe[num] + we[num], :] = rand_color[num]
            tar_per[num, ye[num]:ye[num] + he[num], xe[num]:xe[num] + we[num], :] = 0.0

        return data_per, tar_per

    def beta_func(self, alpha, mixup_num):
        return np.random.beta(alpha, alpha, mixup_num)

    def flip_image(self, image, flip_value):  # flip_value is expected 0 or 1 or 2 or 3
        image_np = image
        # TODO ...
        # if flip_value % 2 == 1:  # 1, 3....flip vertically
        #     image_np = image_np[::-1, :, :]
        if flip_value // 2 == 1:  # 2, 3...flip horizontally
            image_np = image_np[:, ::-1, :]

        return image_np

    def rotate_image(self, image, rotate_value):  # rotate_value is expected 0 or 1 or 2 or 3
        image_np = image
        if rotate_value > 0:  # rotate 90
            image_np = np.rot90(image_np)
        if rotate_value > 1:  # rotate 90
            image_np = np.rot90(image_np)
        if rotate_value > 2:  # rotate 90
            image_np = np.rot90(image_np)
        return image_np

    def normalize_data(self, data):
        # data0_2 = data / 127.5
        # data_norm = data0_2 - 1.0
        data_norm = data / 255.0
        # data_norm = data - 1.0
        return data_norm

    def crop_img(self, data, output_img_W, output_img_H, margin_W, margin_H):
        cropped_img = data.crop((margin_W, margin_H, margin_W + output_img_W, margin_H + output_img_H))
        return cropped_img

    def read_1_data_and_convert_RGB(self, dir, filename_list, extension, width, height):
        images = []
        for filename in filename_list:
            pilIn = Image.open(dir + filename[0] + extension).convert('RGB')
            pilResize = pilIn.resize((width, height))
            image = np.asarray(pilResize)
            image_t = np.transpose(image, (2, 0, 1))
            images.append(image_t)
        return np.asarray(images)

    def write_data_to_img(self, dir, np_arrays, extension):

        for num, np_array in enumerate(np_arrays):
            pil_img = Image.fromarray(np_array)
            pil_img.save(dir + 'debug_' + str(num) + extension)

    def convert_indexInt_to_color_wall(self, data):
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.int32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                d_mod[h][w] = self.wallCrack_color_chan[ele]

        return d_mod

    def convert_indexInt_to_color_for_oneHot(self, data):
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.int32)
        for h, row in enumerate(data):
            for w, col in enumerate(row):
                for c, ele in enumerate(col):
                    if ele == 1.0:
                        d_mod[h][w] = self.cityScape_color_chan[c]

        return d_mod

    def convert_indexInt_to_color_for_oneHot_wallCrack(self, data):
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.int32)
        for h, row in enumerate(data):
            for w, col in enumerate(row):
                for c, ele in enumerate(col):
                    if ele == 1.0:
                        d_mod[h][w] = self.wallCrack_color_chan[c]

        return d_mod


    def make_data_for_1_epoch(self):
        self.image_files_1_epoch = random.sample(self.train_file_list, self.train_file_num)
        self.margin_H = np.random.randint(0, (self.img_height_be_crop - self.img_height + 1), self.train_file_num)
        self.margin_W = np.random.randint(0, (self.img_width_be_crop - self.img_width + 1), self.train_file_num)
        self.flip_list = np.random.randint(0, 4, self.train_file_num)
        self.rotation = np.random.randint(0, 4, self.train_file_num)

        return len(self.image_files_1_epoch)

    def get_data_for_1_batch(self, i, batchsize, train_FLAG=True):
        if train_FLAG == False:
            self.mixup_flag = False
        else:
            self.mixup_flag = True

        data_batch = self.image_files_1_epoch[i:i + batchsize]
        margin_H_batch = self.margin_H[i:i + batchsize]
        margin_W_batch = self.margin_W[i:i + batchsize]
        flip_list_batch = self.flip_list[i:i + batchsize]
        rotate_list_batch = self.rotation[i:i + batchsize]

        images, tars = self.read_data_to_np(self.dir_img, data_batch, self.img_width, self.img_height,
                                            self.img_width_be_crop,
                                            self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag,
                                            self.flip_flag, flip_list_batch, self.rotate_flag, rotate_list_batch,
                                            self.mixup_flag,
                                            self.random_erasing_flag)

        images_n = self.normalize_data(images)

        return images_n, tars

    def make_mask(self, batch_num, img_width, img_height):  # (x255, x1, x1, ..., x1)
        mask = np.ones((batch_num, img_height, img_width, 35), dtype=np.float32)
        mask[:, :, :, 0] = 1.0
        # mask[:, :, :, 2] = 10.0
        # print("mask.shape, ", mask.shape)
        return mask

    def get_data_for_1_batch_val(self, i, batchsize):
        # data_batch = self.val_file_list[i:i + batchsize]
        data_batch = self.file_test_list[i:i + batchsize]

        margin_H_batch = self.margin_H[i:i + batchsize]
        margin_W_batch = self.margin_W[i:i + batchsize]
        flip_list_batch = np.zeros(batchsize, dtype=np.int32)
        rotate_list_batch = np.zeros(batchsize, dtype=np.int32)

        images, tars = self.read_data_to_np(self.dir_img, data_batch, self.img_width, self.img_height,
                                            self.img_width_be_crop,
                                            self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag,
                                            False, flip_list_batch, False, rotate_list_batch, False, False)
        images_n = self.normalize_data(images)
        return images_n, tars

    def get_data_for_1_batch_test(self, i, batchsize):
        data_batch = self.file_test_list[i:i + batchsize]

        margin_H_batch = self.margin_H[i:i + batchsize]
        margin_W_batch = self.margin_W[i:i + batchsize]
        flip_list_batch = np.zeros(batchsize, dtype=np.int32)

        images = self.read_data_to_np(self.dir_test_img, data_batch, self.img_width, self.img_height,
                                      self.img_width_be_crop,
                                      self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, False,
                                      False, flip_list_batch)

        segs = self.read_data_to_np(self.dir_test_seg, data_batch, self.img_width, self.img_height,
                                    self.img_width_be_crop, self.img_height_be_crop, margin_H_batch, margin_W_batch,
                                    self.crop_flag, True,
                                    False, flip_list_batch)

        images_n = self.normalize_data(images)

        return images_n, segs

    def get_data_1_for_prediction(self):
        image = self.read_data_to_np('', [self.predict_img], self.img_width, self.img_height, self.img_width_be_crop,
                                     self.img_height_be_crop, None, None, False, False, False, False)

        image = self.normalize_data(image)

        return image, self.ori_img_width, self.ori_img_height

    def make_img_from_label(self, labels, epoch):  # labels=(first_number, last_number + 1)
        labels_train = self.train_files_1_epoch[labels[0]:labels[1]]
        labels_val = self.list_val_files[labels[0]:labels[1]]
        labels_train_val = labels_train + labels_val
        labels_img_np = self.read_data_to_np_and_convert_RGB(self.SegmentationClass_dir, labels_train_val, '.png',
                                                             self.img_width, self.img_height)
        self.write_data_to_img('debug/label_' + str(epoch) + '_', labels_img_np, '.png')

    def make_img_from_prob(self, probs, epoch):  # probs=(data, height, width)..0-20 value
        # print("probs[0]", probs[0])
        print("probs[0].shape", probs[0].shape)
        probs_RGB = util.convert_indexColor_to_RGB(probs)
        # labels_img_np = self.read_data_to_np_and_convert_RGB(self.SegmentationClass_dir, probs_RGB, '.jpg', self.img_width, self.img_height)
        self.write_data_to_img('debug/prob_' + str(epoch), probs_RGB, '.jpg')

    def get_concat_img_h(self, img1, img2):
        dst = Image.new('RGB', (img1.width + img2.width, img1.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (img1.width, 0))
        return dst

    def get_concat_img_w(self, img1, img2):
        dst = Image.new('RGB', (img1.width, img1.height + img2.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (0, img1.height))
        return dst

    def make_img_from_seg_prob(self, img_X, probsX, segsX, img_Y, probsY, segsY, img_Z, probsZ, segsZ, out_image_dir,
                               epoch, LOG_FILE_NAME):
        probs_segX = []
        for num, prob in enumerate(probsX):
            probs_segX.append(self.convert_indexInt_to_color_wall(prob))
        probs_segX_np = np.array(probs_segX, dtype=np.float32)

        probs_segY = []
        for num, prob in enumerate(probsY):
            probs_segY.append(self.convert_indexInt_to_color_wall(prob))
        probs_segY_np = np.array(probs_segY, dtype=np.float32)

        probs_segZ = []
        for num, prob in enumerate(probsZ):
            probs_segZ.append(self.convert_indexInt_to_color_wall(prob))
        probs_segZ_np = np.array(probs_segZ, dtype=np.float32)

        segX = []
        for num, prob in enumerate(segsX):
            segX.append(self.convert_indexInt_to_color_for_oneHot_wallCrack(prob))
        segX_np = np.array(segX, dtype=np.float32)

        segY = []
        for num, prob in enumerate(segsY):
            segY.append(self.convert_indexInt_to_color_for_oneHot_wallCrack(prob))
        segY_np = np.array(segY, dtype=np.float32)

        segZ = []
        for num, prob in enumerate(segsZ):
            segZ.append(self.convert_indexInt_to_color_for_oneHot_wallCrack(prob))
        segZ_np = np.array(segZ, dtype=np.float32)

        wide_image1 = util.make_output_img_seg(img_X, probs_segX_np, segX_np, out_image_dir, epoch, LOG_FILE_NAME)
        wide_image2 = util.make_output_img_seg(img_Y, probs_segY_np, segY_np, out_image_dir, epoch, LOG_FILE_NAME)
        wide_image3 = util.make_output_img_seg(img_Z, probs_segZ_np, segZ_np, out_image_dir, epoch, LOG_FILE_NAME)

        util.make_output_img_and_save(wide_image1, wide_image2, wide_image3, out_image_dir, epoch, LOG_FILE_NAME)

    def make_img_from_seg_prob_labelme(self, img_X, probsX, segsX, img_Y, probsY, segsY, out_image_dir,
                                       epoch, LOG_FILE_NAME):
        probs_segX = []
        for num, prob in enumerate(probsX):
            probs_segX.append(self.convert_indexInt_to_color_wall(prob))
        probs_segX_np = np.array(probs_segX, dtype=np.float32)

        probs_segY = []
        for num, prob in enumerate(probsY):
            probs_segY.append(self.convert_indexInt_to_color_wall(prob))
        probs_segY_np = np.array(probs_segY, dtype=np.float32)

        segX = []
        for num, prob in enumerate(segsX):
            segX.append(self.convert_indexInt_to_color_for_oneHot_wallCrack(prob))
        segX_np = np.array(segX, dtype=np.float32)

        segY = []
        for num, prob in enumerate(segsY):
            segY.append(self.convert_indexInt_to_color_for_oneHot_wallCrack(prob))
        segY_np = np.array(segY, dtype=np.float32)

        util.make_output_img_seg_labelme(img_X, probs_segX_np, segX_np, img_Y, probs_segY_np, segY_np, out_image_dir,
                                         epoch, LOG_FILE_NAME)
        # wide_image2 = util.make_output_img_seg(img_Y, probs_segY_np, segY_np, out_image_dir, epoch, LOG_FILE_NAME)

        # util.make_output_img_and_save(wide_image1, wide_image2, wide_image3, out_image_dir, epoch, LOG_FILE_NAME)

    def make_img_from_only_prob(self, ori_img_filename, prob, mask_img_filename, width, height, out_image_dir,
                                LOG_FILE_NAME):
        # probs_segX = []
        # for num, prob in enumerate(prob):
        prob_img = self.convert_indexInt_to_color_wall(prob[0])
        prob_img_PIL = Image.fromarray(prob_img.astype(np.uint8))
        prob_img_PIL_resize = prob_img_PIL.resize((width, height))  # not antialius
        prob_img_oriSize = np.asarray(prob_img_PIL_resize)

        # read original image
        ori_img = Image.open(ori_img_filename)

        # read mask image
        mask_img = Image.open(mask_img_filename)
        mask_img_np_01 = np.asarray(mask_img) // 255
        if mask_img_np_01.ndim == 2:
            mask_img_np_01_re = mask_img_np_01.reshape(mask_img_np_01.shape[0], mask_img_np_01.shape[1], 1)
            mask_img_np_01 = np.tile(mask_img_np_01_re, (1, 1, 3))

        # multiply mask, prob
        mul_img = (prob_img_oriSize * mask_img_np_01).astype(np.uint8)
        mul_img_PIL = Image.fromarray(mul_img)

        wide_image = util.make_output_img_mul_mask(mul_img_PIL, ori_img, mask_img, prob_img_PIL_resize)

        wide_image.save(out_image_dir + ori_img_filename + "prob_mask_" + LOG_FILE_NAME + ".png")

    def make_2chanAnno_from_3chanImg(self, np_array):  # class0 -> crack, class1 -> wall
        np_array_1chan = np_array.reshape(np_array.shape[0], np_array.shape[1], 1).astype(np.float32)
        np_array_1chan_re = 1.0 - np_array_1chan
        np_array_2chanAnno = np.concatenate((np_array_1chan, np_array_1chan_re), axis=2)
        # np_array_33chan = np.zeros((np_array.shape[0], np_array.shape[1], 33), dtype=np.float32)
        # np_array_35chanAnno = np.concatenate((np_array_2chanAnno, np_array_33chan), axis=2)

        return np_array_2chanAnno

    def make_4chanAnno_from_3chanImg(self, np_array):  # 0 -> class 0, 1 -> class 1, 2 -> class 2, 3 -> class 3
        np_array_1chan = np_array[:, :, 0].reshape(np_array.shape[0], np_array.shape[1], 1).astype(np.float32)
        np_array_1chan_all1 = np.ones(np_array_1chan.shape, dtype=np.int32)
        np_array_1chan_int = np_array_1chan.astype(np.int32)

        np_array_1chan_N3 = np_array_1chan_int // 3
        np_array_1chan_N23 = np_array_1chan_int // 2
        np_array_1chan_N2 = np_array_1chan_N23 - np_array_1chan_N3
        np_array_1chan_N12 = np_array_1chan_int % 3
        np_array_1chan_N1 = np_array_1chan_N12 - 2 * np_array_1chan_N2
        np_array_1chan_N0 = np_array_1chan_all1 - (np_array_1chan_N1 + np_array_1chan_N2 + np_array_1chan_N3)
        np_array_4chan = np.concatenate((np_array_1chan_N0, np_array_1chan_N1, np_array_1chan_N2, np_array_1chan_N3),
                                        axis=2).astype(np.float32)
        np_array_31chan = np.zeros((np_array.shape[0], np_array.shape[1], 31), dtype=np.float32)
        np_array_35chan = np.concatenate((np_array_4chan, np_array_31chan), axis=2)

        # np_array_1chan_re = 1.0 - np_array_1chan
        # np_array_2canAnno = np.concatenate((np_array_1chan, np_array_1chan_re), axis=2)
        # np_array_33chan = np.zeros((np_array.shape[0], np_array.shape[1], 33), dtype=np.float32)
        # np_array_35chanAnno = np.concatenate((np_array_2canAnno, np_array_33chan), axis=2)

        return np_array_35chan


class Make_datasets_OilLeak(): #seg_dir is used for no-oil image
    def __init__(self, base_dir, img_width, img_height, image_dir, seg_dir, image_val_dir, seg_val_dir,
                 img_width_be_crop, img_height_be_crop,
                 crop_flag=True, val_num=5, flip_flag=True, predict_flag=False, predict_img=None, rotate_flag=True,
                 mixup_flag=False, mixup_rate=1.0, mixup_alpha=0.4, random_erasing_flag=False):

        self.base_dir = base_dir  # /media/webfarmer/HDCZ-UT/dataset/oil_leaks/
        self.test_dir = base_dir  #
        self.img_width = img_width
        self.img_height = img_height
        self.img_width_be_crop = img_width_be_crop
        self.img_height_be_crop = img_height_be_crop
        self.dir_img = base_dir + image_dir  # /media/webfarmer/HDCZ-UT/dataset/oil_leaks/oil_black_puddles_mod_data/
        # self.dir_seg = base_dir + seg_dir  # /media/webfarmer/HDCZ-UT/dataset/oil_leaks/car_oilfree/
        self.dir_test_img = base_dir + image_val_dir  #
        # self.dir_test_seg = base_dir + seg_dir  #
        self.val_num = val_num
        self.predict_flag = predict_flag
        self.predict_img = predict_img
        self.rotate_flag = rotate_flag
        self.crop_flag = crop_flag
        self.flip_flag = flip_flag
        self.mixup_flag = mixup_flag
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha
        self.random_erasing_flag = random_erasing_flag
        file_list_be = self.get_file_names(self.dir_img)
        self.file_list = self.get_only_img_png(file_list_be)
        self.file_list.sort()
        # debug
        # self.print_file_list(self.file_list)
        file_test_list = self.get_file_names(self.dir_test_img)
        print("len(file_test_list), ", len(file_test_list))
        self.file_test_list = self.get_only_img_png(file_test_list)
        self.file_test_list.sort()
        self.wallCrack_color_chan = np.array([
            [255.0, 0.0, 0.0],  # class 0
            [0.0, 0.0, 0.0],  # class 1
            [0.0, 0.0, 0.0],  # class 2
            [0.0, 0.0, 0.0],  # class 3
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]], dtype=np.float32)

        # file_list_seg_be = self.get_file_names(self.dir_seg)
        # self.file_list_seg = self.get_only_png_jpeg_extent(file_list_seg_be)
        # self.file_list_seg.sort()

        self.image_file_num = len(self.file_list)
        self.image_test_file_num = len(self.file_test_list)
        print("self.base_dir = ", self.base_dir)
        print("self.test_dir = ", self.test_dir)
        print("self.dir_img, ", self.dir_img)
        # print("self.dir_seg, ", self.dir_seg)
        print("self.dir_test_img, ", self.dir_test_img)
        # print("self.dir_test_seg, ", self.dir_test_seg)
        print("self.img_width", self.img_width)
        print("self.img_height", self.img_height)
        print("len(self.file_list)", len(self.file_list))

        print("len(self.file_test_list)", len(self.file_test_list))
        print("self.image_file_num", self.image_file_num)
        print("self.image_test_file_num", self.image_test_file_num)
        # print("len(self.file_list_seg), ", len(self.file_list_seg))
        # self.file_list = self.file_list + self.file_list_seg #modefy
        # random.shuffle(self.file_list)
        self.train_file_list = self.file_list[self.val_num:]
        self.val_file_list = self.file_list[:self.val_num]
        # self.train_file_list = self.file_list[:len(self.file_list) - self.val_num]
        # self.val_file_list = self.file_list[len(self.file_list) - self.val_num:]
        self.train_file_num = len(self.train_file_list)
        self.val_file_num = len(self.val_file_list)
        print("self.train_file_num, ", self.train_file_num)
        print("self.val_file_num, ", self.val_file_num)
        print("self.random_erasing_flag, ", self.random_erasing_flag)

    def get_file_names(self, dir_name):
        target_files = []
        for root, dirs, files in os.walk(dir_name):
            targets = [os.path.join(root, f) for f in files]
            target_files.extend(targets)

        return target_files

    def get_only_img(self, list, extension):
        list_mod = []
        for y in list:
            if (y[-9:] == extension):  # only .png
                list_mod.append(y)
        return list_mod

    def get_only_img_png(self, list):
        list_mod = []
        for y in list:
            dir_name, file_name_only = y.rsplit("/", 1)
            if (file_name_only == 'img.png'):  # only .png
                list_mod.append(y)
        return list_mod

    def get_only_png_jpeg_extent(self, list):
        list_mod = []
        for y in list:
            _, extent_only = y.rsplit(".", 1)
            if (extent_only == 'png' or extent_only == 'jpg'):  # only .png
                list_mod.append(y)
        return list_mod


    def print_file_list(self, list):
        for num, filename in enumerate(list):
            dir_name, indi_dir, file_name_only = filename.rsplit("/", 2)
            print(indi_dir)

    def read_data_to_np(self, dir, filename_list, width, height, width_be_crop, height_be_crop, margin_H_batch,
                        margin_W_batch, crop_flag, flip_flag=False, flip_list=None, rotate_flag=False, rotate_list=None,
                        mixup_flag=False):
        images = []
        tars = []
        for num, filename in enumerate(filename_list):
            # if seg_flag:
            #     '''
            #     dir_name, file_name_only = filename.rsplit("/", 1) # ex) .../original_data, xyHdp.jpg
            #     str_base, _ = file_name_only.rsplit(".",1) # ex) xyHdp, jpg
            #     filename_seg = str_base + "_teacher (1).png"
            #     oriIn = Image.open(dir + '/' + filename_seg).convert("RGB")
            #     '''
            dir_name, file_name_only = filename.rsplit("/", 1)  # ex) ...annotation01_180812, img.png
            # str_base, _ = file_name_only.rsplit(".",1) # ex) xyHdp, jpg
            # filename_seg = str_base + "_teacher (1).png"
            tarImg = Image.open(dir_name + '/' + 'label.png')
            # else:
            oriImg = Image.open(filename).convert("RGB")
            self.ori_img_width, self.ori_img_height = oriImg.size

            # if seg_flag:
            if crop_flag:
                before_crop_tar = tarImg.resize((width_be_crop, height_be_crop))
                tar_Resize = self.crop_img(before_crop_tar, width, height, margin_W_batch[num], margin_H_batch[num])
                before_crop_Img = oriImg.resize((width_be_crop, height_be_crop))
                ori_Resize = self.crop_img(before_crop_Img, width, height, margin_W_batch[num], margin_H_batch[num])
            else:
                tar_Resize = tarImg.resize((width, height))
                ori_Resize = oriImg.resize((width, height))
            tar_be = np.asarray(tar_Resize, dtype=np.float32)
            # image = self.make_4chanAnno_from_3chanImg(image_be)
            tar = self.make_2chanAnno_from_1chanImg(tar_be)
            image = np.asarray(ori_Resize, dtype=np.float32)

            if flip_flag:
                tar = self.flip_image(tar, flip_list[num])
                image = self.flip_image(image, flip_list[num])

            if rotate_flag:
                tar = self.rotate_image(tar, rotate_list[num])
                image = self.rotate_image(image, rotate_list[num])

            images.append(image)
            tars.append(tar)
        images = np.asarray(images, dtype=np.float32)
        tars = np.asarray(tars, dtype=np.float32)

        if mixup_flag:
            images, tars = self.do_mixup(images, tars, self.mixup_rate, self.mixup_alpha)

        return images, tars

    def read_2kind_data_to_np(self, dir, filename_list, width, height, width_be_crop, height_be_crop, margin_H_batch,
                        margin_W_batch, crop_flag, flip_flag=False, flip_list=None, rotate_flag=False, rotate_list=None,
                        mixup_flag=False, random_erasing_flag=False):
        images = []
        tars = []
        for num, filename in enumerate(filename_list):
            dir_name, file_name_only = filename.rsplit("/", 1)  # ex) [...tion01_180812, img.png] or [car_oilfree, ...jpg]

            if file_name_only == 'img.png':
                tarImg = Image.open(dir_name + '/' + 'label.png')
                oriImg = Image.open(filename).convert("RGB")
                self.ori_img_width, self.ori_img_height = oriImg.size
            else:
                oriImg = Image.open(filename).convert("RGB")
                self.ori_img_width, self.ori_img_height = oriImg.size
                tarNp = np.zeros((self.ori_img_height, self.ori_img_width), dtype=np.uint8)
                tarImg = Image.fromarray(tarNp)
            # if seg_flag:
            if crop_flag:
                before_crop_tar = tarImg.resize((width_be_crop, height_be_crop))
                tar_Resize = self.crop_img(before_crop_tar, width, height, margin_W_batch[num], margin_H_batch[num])
                before_crop_Img = oriImg.resize((width_be_crop, height_be_crop))
                ori_Resize = self.crop_img(before_crop_Img, width, height, margin_W_batch[num], margin_H_batch[num])
            else:
                tar_Resize = tarImg.resize((width, height))
                ori_Resize = oriImg.resize((width, height))
            tar_be = np.asarray(tar_Resize, dtype=np.float32)
            # image = self.make_4chanAnno_from_3chanImg(image_be)
            tar = self.make_2chanAnno_from_1chanImg(tar_be)
            image = np.asarray(ori_Resize, dtype=np.float32)

            if flip_flag:
                tar = self.flip_image(tar, flip_list[num])
                image = self.flip_image(image, flip_list[num])

            if rotate_flag:
                tar = self.rotate_image(tar, rotate_list[num])
                image = self.rotate_image(image, rotate_list[num])

            images.append(image)
            tars.append(tar)
        images = np.asarray(images, dtype=np.float32)
        tars = np.asarray(tars, dtype=np.float32)

        if mixup_flag:
            images, tars = self.do_mixup(images, tars, self.mixup_rate, self.mixup_alpha)

        if random_erasing_flag:
            images, tars = self.do_random_erasing_for_OIL(images, tars)

        return images, tars


    def do_mixup(self, data, tar, mixup_rate, alpha):  # mixup_rate is expected (0, 1)
        if len(data) < 2:
            return data, tar
        mixup_num = int(len(data) * mixup_rate / 2)
        if mixup_num == 0:
            return data, tar
        data_mixup1 = data[:mixup_num]
        data_mixup2 = data[mixup_num:(mixup_num * 2)]
        data_not_mixup = data[(mixup_num * 2):]
        tar_mixup1 = tar[:mixup_num]
        tar_mixup2 = tar[mixup_num:(mixup_num * 2)]
        tar_not_mixup = tar[(mixup_num * 2):]
        lam = self.beta_func(alpha, mixup_num).reshape(-1, 1, 1, 1)
        mixuped_data = lam * data_mixup1 + (1 - lam) * data_mixup2  # do mixup
        mixuped_tar = lam * tar_mixup1 + (1 - lam) * tar_mixup2  # do mixup
        data_con = np.concatenate((mixuped_data, data_not_mixup), axis=0)
        tar_con = np.concatenate((mixuped_tar, tar_not_mixup), axis=0)
        return data_con, tar_con

    def do_random_erasing(self, data, tar, prob=0.5, sl=0.02, sh=0.4, r1=0.3, r2=0.7):  # random erasing implementation
        do_R_E_num = int(len(data) * prob)
        random_per = np.random.randint(0, len(data), len(data))
        data_per = data[random_per]
        tar_per = tar[random_per]

        # get variants
        re_0_1 = np.random.rand(do_R_E_num)
        re = re_0_1 * (r2 - r1) + r1
        se_0_1 = np.random.rand(do_R_E_num)
        se = (data.shape[1] * data.shape[2]) * (sl + se_0_1 * (sh - sl))
        he = ((se * re) ** 0.5).astype(np.int32)
        we = ((se / re) ** 0.5).astype(np.int32)
        margin_h = data.shape[1] - he
        margin_w = data.shape[2] - we
        margin_rand = np.random.rand(do_R_E_num, 2)

        xe = (margin_w * margin_rand[:, 0]).astype(np.int32)
        ye = (margin_h * margin_rand[:, 1]).astype(np.int32)

        rand_color = np.random.randint(0, 255, (do_R_E_num, 3))

        for num, data1, in enumerate(data_per):
            data1[ye:ye + he, xe:xe + we] = rand_color[num]
            tar_per[ye:ye + he, xe:xe + we, :] = 0.0

        return data_per, tar_per


    def do_random_erasing_for_OIL(self, data, tar, prob=0.5, sl=0.02, sh=0.4, r1=0.3, r2=0.7):  # random erasing implementation
        # avoid the case that erasing window is similar to oil...so avoid black or gray color
        do_R_E_num = int(len(data) * prob)
        np_arange = np.arange(len(data))
        random_per = np.random.permutation(np_arange)
        data_per = data[random_per]
        tar_per = tar[random_per]

        # get variants
        re_0_1 = np.random.rand(do_R_E_num)
        re = re_0_1 * (r2 - r1) + r1
        se_0_1 = np.random.rand(do_R_E_num)
        se = (data.shape[1] * data.shape[2]) * (sl + se_0_1 * (sh - sl))
        he = ((se * re) ** 0.5).astype(np.int32)
        we = ((se / re) ** 0.5).astype(np.int32)
        margin_h = data.shape[1] - he
        margin_w = data.shape[2] - we
        margin_rand = np.random.rand(do_R_E_num, 2)

        xe = (margin_w * margin_rand[:, 0]).astype(np.int32)
        ye = (margin_h * margin_rand[:, 1]).astype(np.int32)
        #TODO ... modified point
        rand_color = np.random.randint(128, 255, (do_R_E_num, 3))

        # for num, data1, in enumerate(data_per):
        #     data1[ye:ye + he, xe:xe + we] = rand_color[num]
        #     tar_per[ye:ye + he, xe:xe + we, :] = 0.0
        for num, data1, in enumerate(data_per):
            if do_R_E_num <= num:
                break
            data1[ye[num]:ye[num]+he[num], xe[num]:xe[num]+we[num], :] = rand_color[num]
            tar_per[num, ye[num]:ye[num]+he[num], xe[num]:xe[num]+we[num], :] = 0.0

        return data_per, tar_per



    def beta_func(self, alpha, mixup_num):
        return np.random.beta(alpha, alpha, mixup_num)

    def flip_image(self, image, flip_value):  # flip_value is expected 0 or 1 or 2 or 3
        image_np = image
        if flip_value % 2 == 1:  # 1, 3....flip vertically
            image_np = image_np[::-1, :, :]
        if flip_value // 2 == 1:  # 2, 3...flip horizontally
            image_np = image_np[:, ::-1, :]
        return image_np

    def rotate_image(self, image, rotate_value):  # rotate_value is expected 0 or 1 or 2 or 3
        image_np = image
        if rotate_value > 0:  # rotate 90
            image_np = np.rot90(image_np)
        if rotate_value > 1:  # rotate 90
            image_np = np.rot90(image_np)
        if rotate_value > 2:  # rotate 90
            image_np = np.rot90(image_np)
        return image_np

    def normalize_data(self, data):
        # data0_2 = data / 127.5
        # data_norm = data0_2 - 1.0
        data_norm = data / 255.0
        # data_norm = data - 1.0
        return data_norm

    def crop_img(self, data, output_img_W, output_img_H, margin_W, margin_H):
        cropped_img = data.crop((margin_W, margin_H, margin_W + output_img_W, margin_H + output_img_H))
        return cropped_img

    def read_1_data_and_convert_RGB(self, dir, filename_list, extension, width, height):
        images = []
        for filename in filename_list:
            pilIn = Image.open(dir + filename[0] + extension).convert('RGB')
            pilResize = pilIn.resize((width, height))
            image = np.asarray(pilResize)
            image_t = np.transpose(image, (2, 0, 1))
            images.append(image_t)
        return np.asarray(images)

    def write_data_to_img(self, dir, np_arrays, extension):

        for num, np_array in enumerate(np_arrays):
            pil_img = Image.fromarray(np_array)
            pil_img.save(dir + 'debug_' + str(num) + extension)

    def convert_color_to_30chan(self, data):  # for cityScape dataset when use Tensorflow
        d_mod = np.zeros((data.shape[0], data.shape[1], 30), dtype=np.float32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                for num, chan in enumerate(self.cityScape_color_chan):
                    if np.allclose(chan, ele):
                        d_mod[h][w][num] = 1.0
        return d_mod

    def convert_30chan_to_color(self, data):
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.float32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                for num, chan in enumerate(ele):
                    if chan == 1.0:
                        d_mod[h][w] = self.cityScape_color_chan[num]
        return d_mod

    def convert_color_to_indexInt(self, data):  # for cityScape dataset when use Chainer
        d_mod = np.zeros((data.shape[0], data.shape[1]), dtype=np.int32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                for num, chan in enumerate(self.cityScape_color_chan):
                    if np.allclose(chan, ele):
                        d_mod[h][w] = num
        return d_mod

    def convert_indexInt_to_color(self, data):
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.int32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                d_mod[h][w] = self.cityScape_color_chan[ele]

        return d_mod

    def convert_indexInt_to_color_wall(self, data):
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.int32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                d_mod[h][w] = self.wallCrack_color_chan[ele]

        return d_mod

    def convert_indexInt_to_color_for_oneHot(self, data):
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.int32)
        for h, row in enumerate(data):
            for w, col in enumerate(row):
                for c, ele in enumerate(col):
                    if ele == 1.0:
                        d_mod[h][w] = self.cityScape_color_chan[c]

        return d_mod

    def convert_indexInt_to_color_for_oneHot_wallCrack(self, data):
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.int32)
        for h, row in enumerate(data):
            for w, col in enumerate(row):
                for c, ele in enumerate(col):
                    if ele == 1.0:
                        d_mod[h][w] = self.wallCrack_color_chan[c]

        return d_mod

    def convert_to_0_1_class_(self, d):
        d_mod = np.zeros((d.shape[0], d.shape[1], d.shape[2], self.class_num), dtype=np.float32)

        for num, image1 in enumerate(d):
            for h, row in enumerate(image1):
                for w, ele in enumerate(row):
                    if int(ele) == 255:  # border
                        continue
                    # d_mod[num][h][w][int(ele) - 1] = 1.0
                    d_mod[num][h][w][int(ele)] = 1.0
        return d_mod

    def make_data_for_1_epoch(self):
        self.image_files_1_epoch = random.sample(self.train_file_list, self.train_file_num)
        self.margin_H = np.random.randint(0, (self.img_height_be_crop - self.img_height + 1), self.train_file_num)
        self.margin_W = np.random.randint(0, (self.img_width_be_crop - self.img_width + 1), self.train_file_num)
        self.flip_list = np.random.randint(0, 4, self.train_file_num)
        self.rotation = np.random.randint(0, 4, self.train_file_num)
        return len(self.image_files_1_epoch)


    def get_data_for_1_batch(self, i, batchsize, train_FLAG=True):
        data_batch = self.image_files_1_epoch[i:i + batchsize]
        margin_H_batch = self.margin_H[i:i + batchsize]
        margin_W_batch = self.margin_W[i:i + batchsize]
        flip_list_batch = self.flip_list[i:i + batchsize]
        rotate_list_batch = self.rotation[i:i + batchsize]

        images, tars = self.read_2kind_data_to_np(self.dir_img, data_batch, self.img_width, self.img_height,
                                            self.img_width_be_crop,
                                            self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag,
                                            self.flip_flag, flip_list_batch, self.rotate_flag, rotate_list_batch,
                                            self.mixup_flag, self.random_erasing_flag)

        images_n = self.normalize_data(images)

        return images_n, tars

    def make_mask(self, batch_num, img_width, img_height):  # (x255, x1, x1, ..., x1)
        mask = np.ones((batch_num, img_height, img_width, 35), dtype=np.float32)
        mask[:, :, :, 0] = 1.0
        # mask[:, :, :, 2] = 10.0
        # print("mask.shape, ", mask.shape)
        return mask

    def get_data_for_1_batch_val(self, i, batchsize):
        # data_batch = self.val_file_list[i:i + batchsize]
        data_batch = self.file_test_list[i:i + batchsize]
        margin_H_batch = self.margin_H[i:i + batchsize]
        margin_W_batch = self.margin_W[i:i + batchsize]
        flip_list_batch = np.zeros(batchsize, dtype=np.int32)
        rotate_list_batch = np.zeros(batchsize, dtype=np.int32)

        images, tars = self.read_2kind_data_to_np(self.dir_img, data_batch, self.img_width, self.img_height,
                                            self.img_width_be_crop,
                                            self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag,
                                            False, flip_list_batch, False, rotate_list_batch, False, False)

        images_n = self.normalize_data(images)

        return images_n, tars

    def get_data_for_1_batch_test(self, i, batchsize):
        data_batch = self.file_test_list[i:i + batchsize]

        margin_H_batch = self.margin_H[i:i + batchsize]
        margin_W_batch = self.margin_W[i:i + batchsize]
        flip_list_batch = np.zeros(batchsize, dtype=np.int32)

        images = self.read_data_to_np(self.dir_test_img, data_batch, self.img_width, self.img_height,
                                      self.img_width_be_crop,
                                      self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, False,
                                      False, flip_list_batch)

        segs = self.read_data_to_np(self.dir_test_seg, data_batch, self.img_width, self.img_height,
                                    self.img_width_be_crop, self.img_height_be_crop, margin_H_batch, margin_W_batch,
                                    self.crop_flag, True,
                                    False, flip_list_batch)

        images_n = self.normalize_data(images)

        return images_n, segs

    def get_data_1_for_prediction(self):
        image = self.read_data_to_np('', [self.predict_img], self.img_width, self.img_height, self.img_width_be_crop,
                                     self.img_height_be_crop, None, None, False, False, False, False)

        image = self.normalize_data(image)

        return image, self.ori_img_width, self.ori_img_height

    def make_img_from_label(self, labels, epoch):  # labels=(first_number, last_number + 1)
        labels_train = self.train_files_1_epoch[labels[0]:labels[1]]
        labels_val = self.list_val_files[labels[0]:labels[1]]
        labels_train_val = labels_train + labels_val
        labels_img_np = self.read_data_to_np_and_convert_RGB(self.SegmentationClass_dir, labels_train_val, '.png',
                                                             self.img_width, self.img_height)
        self.write_data_to_img('debug/label_' + str(epoch) + '_', labels_img_np, '.png')

    def make_img_from_prob(self, probs, epoch):  # probs=(data, height, width)..0-20 value
        # print("probs[0]", probs[0])
        print("probs[0].shape", probs[0].shape)
        probs_RGB = util.convert_indexColor_to_RGB(probs)
        # labels_img_np = self.read_data_to_np_and_convert_RGB(self.SegmentationClass_dir, probs_RGB, '.jpg', self.img_width, self.img_height)
        self.write_data_to_img('debug/prob_' + str(epoch), probs_RGB, '.jpg')

    def get_concat_img_h(self, img1, img2):
        dst = Image.new('RGB', (img1.width + img2.width, img1.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (img1.width, 0))
        return dst

    def get_concat_img_w(self, img1, img2):
        dst = Image.new('RGB', (img1.width, img1.height + img2.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (0, img1.height))
        return dst

    def make_img_from_seg_prob(self, img_X, probsX, segsX, img_Y, probsY, segsY, img_Z, probsZ, segsZ, out_image_dir,
                               epoch, LOG_FILE_NAME):
        probs_segX = []
        for num, prob in enumerate(probsX):
            probs_segX.append(self.convert_indexInt_to_color_wall(prob))
        probs_segX_np = np.array(probs_segX, dtype=np.float32)

        probs_segY = []
        for num, prob in enumerate(probsY):
            probs_segY.append(self.convert_indexInt_to_color_wall(prob))
        probs_segY_np = np.array(probs_segY, dtype=np.float32)

        probs_segZ = []
        for num, prob in enumerate(probsZ):
            probs_segZ.append(self.convert_indexInt_to_color_wall(prob))
        probs_segZ_np = np.array(probs_segZ, dtype=np.float32)

        segX = []
        for num, prob in enumerate(segsX):
            segX.append(self.convert_indexInt_to_color_for_oneHot_wallCrack(prob))
        segX_np = np.array(segX, dtype=np.float32)

        segY = []
        for num, prob in enumerate(segsY):
            segY.append(self.convert_indexInt_to_color_for_oneHot_wallCrack(prob))
        segY_np = np.array(segY, dtype=np.float32)

        segZ = []
        for num, prob in enumerate(segsZ):
            segZ.append(self.convert_indexInt_to_color_for_oneHot_wallCrack(prob))
        segZ_np = np.array(segZ, dtype=np.float32)

        wide_image1 = util.make_output_img_seg(img_X, probs_segX_np, segX_np, out_image_dir, epoch, LOG_FILE_NAME)
        wide_image2 = util.make_output_img_seg(img_Y, probs_segY_np, segY_np, out_image_dir, epoch, LOG_FILE_NAME)
        wide_image3 = util.make_output_img_seg(img_Z, probs_segZ_np, segZ_np, out_image_dir, epoch, LOG_FILE_NAME)

        util.make_output_img_and_save(wide_image1, wide_image2, wide_image3, out_image_dir, epoch, LOG_FILE_NAME)

    def make_img_from_seg_prob_labelme(self, img_X, probsX, segsX, img_Y, probsY, segsY, out_image_dir,
                                       epoch, LOG_FILE_NAME):
        probs_segX = []
        for num, prob in enumerate(probsX):
            probs_segX.append(self.convert_indexInt_to_color_wall(prob))
        probs_segX_np = np.array(probs_segX, dtype=np.float32)

        probs_segY = []
        for num, prob in enumerate(probsY):
            probs_segY.append(self.convert_indexInt_to_color_wall(prob))
        probs_segY_np = np.array(probs_segY, dtype=np.float32)

        segX = []
        for num, prob in enumerate(segsX):
            segX.append(self.convert_indexInt_to_color_for_oneHot_wallCrack(prob))
        segX_np = np.array(segX, dtype=np.float32)

        segY = []
        for num, prob in enumerate(segsY):
            segY.append(self.convert_indexInt_to_color_for_oneHot_wallCrack(prob))
        segY_np = np.array(segY, dtype=np.float32)

        util.make_output_img_seg_labelme(img_X, probs_segX_np, segX_np, img_Y, probs_segY_np, segY_np, out_image_dir,
                                         epoch, LOG_FILE_NAME)
        # wide_image2 = util.make_output_img_seg(img_Y, probs_segY_np, segY_np, out_image_dir, epoch, LOG_FILE_NAME)

        # util.make_output_img_and_save(wide_image1, wide_image2, wide_image3, out_image_dir, epoch, LOG_FILE_NAME)

    def make_img_from_only_prob(self, ori_img_filename, prob, mask_img_filename, width, height, out_image_dir,
                                LOG_FILE_NAME):
        # probs_segX = []
        # for num, prob in enumerate(prob):
        prob_img = self.convert_indexInt_to_color_wall(prob[0])
        prob_img_PIL = Image.fromarray(prob_img.astype(np.uint8))
        prob_img_PIL_resize = prob_img_PIL.resize((width, height))  # not antialius
        prob_img_oriSize = np.asarray(prob_img_PIL_resize)

        # read original image
        ori_img = Image.open(ori_img_filename)

        # read mask image
        mask_img = Image.open(mask_img_filename)
        mask_img_np_01 = np.asarray(mask_img) // 255
        if mask_img_np_01.ndim == 2:
            mask_img_np_01_re = mask_img_np_01.reshape(mask_img_np_01.shape[0], mask_img_np_01.shape[1], 1)
            mask_img_np_01 = np.tile(mask_img_np_01_re, (1, 1, 3))

        # multiply mask, prob
        mul_img = (prob_img_oriSize * mask_img_np_01).astype(np.uint8)
        mul_img_PIL = Image.fromarray(mul_img)

        wide_image = util.make_output_img_mul_mask(mul_img_PIL, ori_img, mask_img, prob_img_PIL_resize)

        wide_image.save(out_image_dir + ori_img_filename + "prob_mask_" + LOG_FILE_NAME + ".png")

    def make_2chanAnno_from_3chanImg(self, np_array):  # class0 -> crack, class1 -> wall
        np_array_1chan = np_array.reshape(np_array.shape[0], np_array.shape[1], 1).astype(np.float32)
        np_array_1chan_re = 1.0 - np_array_1chan
        np_array_2canAnno = np.concatenate((np_array_1chan, np_array_1chan_re), axis=2)
        np_array_33chan = np.zeros((np_array.shape[0], np_array.shape[1], 33), dtype=np.float32)
        np_array_35chanAnno = np.concatenate((np_array_2canAnno, np_array_33chan), axis=2)

        return np_array_35chanAnno

    def make_2chanAnno_from_1chanImg(self, np_array):  # class0 -> crack, class1 -> wall
        np_array_1chan = np_array.reshape(np_array.shape[0], np_array.shape[1], 1).astype(np.float32)
        np_array_1chan_re = 1.0 - np_array_1chan
        np_array_2chanAnno = np.concatenate((np_array_1chan, np_array_1chan_re), axis=2)
        # np_array_33chan = np.zeros((np_array.shape[0], np_array.shape[1], 33), dtype=np.float32)
        # np_array_35chanAnno = np.concatenate((np_array_2canAnno, np_array_33chan), axis=2)

        return np_array_2chanAnno

    def make_4chanAnno_from_3chanImg(self, np_array):  # 0 -> class 0, 1 -> class 1, 2 -> class 2, 3 -> class 3
        np_array_1chan = np_array[:, :, 0].reshape(np_array.shape[0], np_array.shape[1], 1).astype(np.float32)
        np_array_1chan_all1 = np.ones(np_array_1chan.shape, dtype=np.int32)
        np_array_1chan_int = np_array_1chan.astype(np.int32)

        np_array_1chan_N3 = np_array_1chan_int // 3
        np_array_1chan_N23 = np_array_1chan_int // 2
        np_array_1chan_N2 = np_array_1chan_N23 - np_array_1chan_N3
        np_array_1chan_N12 = np_array_1chan_int % 3
        np_array_1chan_N1 = np_array_1chan_N12 - 2 * np_array_1chan_N2
        np_array_1chan_N0 = np_array_1chan_all1 - (np_array_1chan_N1 + np_array_1chan_N2 + np_array_1chan_N3)
        np_array_4chan = np.concatenate((np_array_1chan_N0, np_array_1chan_N1, np_array_1chan_N2, np_array_1chan_N3),
                                        axis=2).astype(np.float32)
        np_array_31chan = np.zeros((np_array.shape[0], np_array.shape[1], 31), dtype=np.float32)
        np_array_35chan = np.concatenate((np_array_4chan, np_array_31chan), axis=2)

        # np_array_1chan_re = 1.0 - np_array_1chan
        # np_array_2canAnno = np.concatenate((np_array_1chan, np_array_1chan_re), axis=2)
        # np_array_33chan = np.zeros((np_array.shape[0], np_array.shape[1], 33), dtype=np.float32)
        # np_array_35chanAnno = np.concatenate((np_array_2canAnno, np_array_33chan), axis=2)

        return np_array_35chan


def check_data(filename):
    mnist_npz = np.load(filename)
    print("type(mnist_npz), ", type(mnist_npz))
    print("mnist_npz.keys(), ", mnist_npz.keys())
    print("mnist_npz['x_train'].shape, ", mnist_npz['x_train'].shape)
    print("mnist_npz['x_test'].shape, ", mnist_npz['x_test'].shape)
    print("mnist_npz['x_valid'].shape, ", mnist_npz['x_valid'].shape)
    print("mnist_npz['y_train'].shape, ", mnist_npz['y_train'].shape)
    print("mnist_npz['y_test'].shape, ", mnist_npz['y_test'].shape)
    print("mnist_npz['y_valid'].shape, ", mnist_npz['y_valid'].shape)




if __name__ == '__main__':
    #debug
    # FILE_NAME = './mnist.npz'
    # check_mnist_npz(FILE_NAME)
    # base_dir = '/media/webfarmer/FourTBData/datasets/wall/'
    # image_dir = 'wall_only_img/'
    # img_dir2 = 'wall_with_crack_only_img/'
    # make_datasets = Make_datasets_AE(base_dir, image_dir, img_dir2, 56, 56, 74, 74,
    #              crop_flag=False, val_num=5, flip_flag=False, predict_flag=False, predict_img=None)
    #
    # make_datasets.make_data_for_1_epoch()
    # images = make_datasets.get_data_for_1_batch(0, 10)
    # print("images.shape, ", images.shape)
    # print("np.max(images), ", np.max(images))
    # print("np.min(images), ", np.min(images))
    base_dir = '/media/webfarmer/FourTBData/datasets/CityScape/'
    image_dir = 'train/image/'
    seg_dir = 'train/mask/'
    image_val_dir = 'test/image/'
    seg_val_dir = 'test/mask/'

    make_datasets = Make_datasets_CityScape(base_dir, 128, 128, image_dir, seg_dir, image_val_dir, seg_val_dir,
                 256, 128, crop_flag=False)

    make_datasets.make_data_for_1_epoch()
    images, segs = make_datasets.get_data_for_1_batch_val(0, 10)
    print("images.shape, ", images.shape)
    print("segs.shape, ", segs.shape)
    print("np.max(images), ", np.max(images))
    print("np.min(images), ", np.min(images))
    print("np.max(segs), ", np.max(segs))
    print("np.min(segs), ", np.min(segs))
