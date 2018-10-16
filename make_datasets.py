import numpy as np
import os
import random
from PIL import Image
import cv2

class Make_datasets_CityScape():
    def __init__(self, base_dir, img_width, img_height, image_dir, seg_dir, image_val_dir, seg_val_dir,
                 img_width_be_crop, img_height_be_crop, crop_flag=False):
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


    def convert_color_to_30chan(self, data): # for cityScape dataset when use Tensorflow
        # print("data.shape", data.shape)
        # print("self.cityScape_color_chan.shape", self.cityScape_color_chan.shape)
        d_mod = np.zeros((data.shape[0], data.shape[1], 30), dtype=np.float32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                for num, chan in enumerate(self.cityScape_color_chan):
                    # print("ele.shape", ele.shape)
                    # print("chan.shape", chan.shape)
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


    def convert_color_to_indexInt(self, data): # for cityScape dataset when use Chainer
        d_mod = np.zeros((data.shape[0], data.shape[1]), dtype=np.int32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                for num, chan in enumerate(self.cityScape_color_chan):
                    if np.allclose(chan, ele):
                        d_mod[h][w]= num
        return d_mod


    def convert_indexInt_to_color(self, data):
        # print("data.shape", data.shape)
        # print("data[0][0]", data[0][0])
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.int32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                # print("ele", ele)
                # print("type(ele), ", type(ele))
                # ele_np = cuda.to_cpu(ele)
                # print("type(ele_np)", type(ele_np))
                # print("self.cityScape_color_chan[ele]", self.cityScape_color_chan[ele])
                d_mod[h][w] = self.cityScape_color_chan[ele]

                # d_mod[h][w][0] = self.cityScape_color_chan[ele][0]
                # d_mod[h][w][1] = self.cityScape_color_chan[ele][1]
                # d_mod[h][w][2] = self.cityScape_color_chan[ele][2]

        return d_mod


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


    def convert_to_0_1_class_(self, d):
        d_mod = np.zeros((d.shape[0], d.shape[1], d.shape[2], self.class_num), dtype=np.float32)

        for num, image1 in enumerate(d):
            for h, row in enumerate(image1):
                for w, ele in enumerate(row):
                    if int(ele) == 255:#border
                    # if int(ele) == 255 or int(ele) == 0:#border and backgrounds
                        # d_mod[num][h][w][20] = 1.0
                        continue
                    # d_mod[num][h][w][int(ele) - 1] = 1.0
                    d_mod[num][h][w][int(ele)] = 1.0
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

        # labels_0_1 = self.convert_to_0_1_class_(labels)
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

        # labels_0_1 = self.convert_to_0_1_class_(labels)
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
