import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from imageio import imread
from PIL import Image
import cv2
import os
from scipy.io import loadmat
from tqdm import tqdm
from torch.nn import functional as F
import os

import cv2
import numpy as np
import random

import torch
from torch.nn import functional as F
from torch.utils import data

from config import config_hrnet_v2 as config


class UWFSDataLoader(torch.utils.data.Dataset):
    def __init__(self, output_image_height=700, images=None,
                 masks=None, normalizer=None, channel_values=None):
        self.output_image_height = output_image_height
        self.images = images
        self.masks = masks
        self.normalizer = normalizer
        self.channel_values = channel_values

        if not self.channel_values:
            self.label_dictionary = {
                0:  {'name': 'unlabeled',   'train_id': 255, 'color': (0,   0,   0)},
                1:  {'name': 'crab',        'train_id': 0,   'color': (128, 64,  128)},
                2:  {'name': 'crocodile',   'train_id': 1,   'color': (244, 35,  232)},
                3:  {'name': 'dolphin',     'train_id': 2,   'color': (70,  70,  70)},
                4:  {'name': 'frog',        'train_id': 3,   'color': (102, 102, 156)},
                5:  {'name': 'nettles',     'train_id': 4,   'color': (190, 153, 153)},
                6:  {'name': 'octopus',     'train_id': 5,   'color': (153, 153, 153)},
                7:  {'name': 'otter',       'train_id': 6,   'color': (250, 170, 30)},
                8:  {'name': 'penguin',     'train_id': 7,   'color': (220, 220, 0)},
                9:  {'name': 'polar_bear',  'train_id': 8,   'color': (107, 142, 35)},
                10: {'name': 'sea_anemone', 'train_id': 9,  'color': (152, 251, 152)},
                11: {'name': 'sea_urchin',  'train_id': 10,  'color': (70,  130, 180)},
                12: {'name': 'seahorse',    'train_id': 11,  'color': (220, 20,  60)},
                13: {'name': 'seal',        'train_id': 12,  'color': (253, 0,   0)},
                14: {'name': 'shark',       'train_id': 13,  'color': (0,   0,   142)},
                15: {'name': 'shrimp',      'train_id': 14,  'color': (0,   0,   70)},
                16: {'name': 'star_fish',   'train_id': 15,  'color': (0,   60,  100)},
                17: {'name': 'stingray',    'train_id': 16,  'color': (0,   80,  100)},
                18: {'name': 'squid',       'train_id': 17,  'color': (0,   0,   230)},
                19: {'name': 'turtle',      'train_id': 18,  'color': (119, 11,  32)},
                20: {'name': 'whale',       'train_id': 19,  'color': (111, 74,  0)},
                21: {'name': 'nudibranch',  'train_id': 20,  'color': (81,  0,   81)},
                22: {'name': 'coral',       'train_id': 21,  'color': (250, 170, 160)},
                23: {'name': 'rock',        'train_id': 22,  'color': (230, 150, 140)},
                24: {'name': 'water',       'train_id': 23,  'color': (180, 165, 180)},
                25: {'name': 'sand',        'train_id': 24,  'color': (150, 100, 100)},
                26: {'name': 'plant',       'train_id': 25,  'color': (150, 120, 90)},
                27: {'name': 'human',       'train_id': 26,  'color': (153, 153, 153)},
                28: {'name': 'reef',        'train_id': 27,  'color': (0,   0,   110)},
                29: {'name': 'others',      'train_id': 28,  'color': (47,  220, 70)}
            }
        else:
            self.label_dictionary = self.channel_values

        self.length = len(self.images)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_label_dict(self):
        return self.label_dictionary

    def get_image_nd_label(self, index):
        image = self.images[index]
        label = self.masks[index]
        # print(np.unique(label))
        # img_h, img_w = image.shape[0], image.shape[1]
        # ratio = img_w / img_h
        # out_width = int(ratio * self.output_image_height)
        # output_size= (self.output_image_height, out_width)
        # output_size = (self.output_image_height, self.output_image_height)
        # if (image.shape[0], image.shape[1]) != output_size:
        #     image = cv2.resize(image, output_size, 0, 0, interpolation=cv2.INTER_LINEAR)
        # if (label.shape[0], label.shape[1]) != output_size:
        #     label = cv2.resize(label, output_size, 0, 0, interpolation=cv2.INTER_NEAREST)
        # print(image.shape, label.shape)
        # print(np.unique(label))
        return image, label

    def __getitem__(self, index):
        img, label_image_gray = self.get_image_nd_label(index)

        if self.normalizer:
            image, label_image_gray = self.normalizer(img, label_image_gray)
        else:
            raise NotImplementedError("Normalizer not implemented...")

        return image, label_image_gray

    def __len__(self):
        return len(self.images)


class UWFSDataLoaderVal(torch.utils.data.Dataset):
    def __init__(self, base_size, crop_size, multi_scale, output_image_height=700, images=None,
                 masks=None, normalizer=None, channel_values=None):
        self.output_image_height = output_image_height
        self.images = images
        self.masks = masks
        self.normalizer = normalizer
        self.channel_values = channel_values

        self.multi_scale = multi_scale
        self.flip = False
        self.crop_size = crop_size

        self.base_size = base_size
        self.crop_size = crop_size
        # self.ignore_label = 255

        # self.mean = mean
        # self.std = std
        # self.scale_factor = scale_factor
        # self.downsample_rate = 1. / downsample_rate

        if not self.channel_values:
            self.label_dictionary = {
                0:  {'name': 'unlabeled',   'train_id': 255, 'color': (0,   0,   0)},
                1:  {'name': 'crab',        'train_id': 0,   'color': (128, 64,  128)},
                2:  {'name': 'crocodile',   'train_id': 1,   'color': (244, 35,  232)},
                3:  {'name': 'dolphin',     'train_id': 2,   'color': (70,  70,  70)},
                4:  {'name': 'frog',        'train_id': 3,   'color': (102, 102, 156)},
                5:  {'name': 'nettles',     'train_id': 4,   'color': (190, 153, 153)},
                6:  {'name': 'octopus',     'train_id': 5,   'color': (153, 153, 153)},
                7:  {'name': 'otter',       'train_id': 6,   'color': (250, 170, 30)},
                8:  {'name': 'penguin',     'train_id': 7,   'color': (220, 220, 0)},
                9:  {'name': 'polar_bear',  'train_id': 8,   'color': (107, 142, 35)},
                10: {'name': 'sea_anemone', 'train_id': 9,  'color': (152, 251, 152)},
                11: {'name': 'sea_urchin',  'train_id': 10,  'color': (70,  130, 180)},
                12: {'name': 'seahorse',    'train_id': 11,  'color': (220, 20,  60)},
                13: {'name': 'seal',        'train_id': 12,  'color': (253, 0,   0)},
                14: {'name': 'shark',       'train_id': 13,  'color': (0,   0,   142)},
                15: {'name': 'shrimp',      'train_id': 14,  'color': (0,   0,   70)},
                16: {'name': 'star_fish',   'train_id': 15,  'color': (0,   60,  100)},
                17: {'name': 'stingray',    'train_id': 16,  'color': (0,   80,  100)},
                18: {'name': 'squid',       'train_id': 17,  'color': (0,   0,   230)},
                19: {'name': 'turtle',      'train_id': 18,  'color': (119, 11,  32)},
                20: {'name': 'whale',       'train_id': 19,  'color': (111, 74,  0)},
                21: {'name': 'nudibranch',  'train_id': 20,  'color': (81,  0,   81)},
                22: {'name': 'coral',       'train_id': 21,  'color': (250, 170, 160)},
                23: {'name': 'rock',        'train_id': 22,  'color': (230, 150, 140)},
                24: {'name': 'water',       'train_id': 23,  'color': (180, 165, 180)},
                25: {'name': 'sand',        'train_id': 24,  'color': (150, 100, 100)},
                26: {'name': 'plant',       'train_id': 25,  'color': (150, 120, 90)},
                27: {'name': 'human',       'train_id': 26,  'color': (153, 153, 153)},
                28: {'name': 'reef',        'train_id': 27,  'color': (0,   0,   110)},
                29: {'name': 'others',      'train_id': 28,  'color': (47,  220, 70)}
            }
        else:
            self.label_dictionary = self.channel_values

        self.num_classes = len(list(self.label_dictionary.keys())) - 1

        self.length = len(self.images)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_label_dict(self):
        return self.label_dictionary

    def get_image_nd_label(self, index):
        image = self.images[index]
        label = self.masks[index]
        # print(np.unique(label))
        # img_h, img_w = image.shape[0], image.shape[1]
        # ratio = img_w / img_h
        # out_width = int(ratio * self.output_image_height)
        # output_size= (self.output_image_height, out_width)
        # output_size = (self.output_image_height, self.output_image_height)
        # if (image.shape[0], image.shape[1]) != output_size:
        #     image = cv2.resize(image, output_size, 0, 0, interpolation=cv2.INTER_LINEAR)
        # if (label.shape[0], label.shape[1]) != output_size:
        #     label = cv2.resize(label, output_size, 0, 0, interpolation=cv2.INTER_NEAREST)
        # print(image.shape, label.shape)
        # print(np.unique(label))
        return image, label

    def __getitem__(self, index):
        img, label_image_gray = self.get_image_nd_label(index)

        size = img.shape

        if self.normalizer:
            image, label_image_gray = self.normalizer(img, label_image_gray)
        else:
            raise NotImplementedError("Normalizer not implemented...")

        return image, label_image_gray, np.array(size), f'{index}.png'

    def __len__(self):
        return len(self.images)

    def convert_label(self, label, inverse=False):
        label_image_gray = label.copy()

        return label_image_gray

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width]).cuda()
        for scale in scales:
            new_img = image  # self.multi_scale_aug(image=image,
                                           # rand_scale=scale,
                                           # rand_crop=False)
            height, width = new_img.shape[:-1]

            new_img = new_img.transpose((2, 0, 1))
            new_img = np.expand_dims(new_img, axis=0)
            new_img = torch.from_numpy(new_img)
            preds = self.inference(config, model, new_img, False)
            preds = preds[:, :, 0:height, 0:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width),
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))

    # def input_transform(self, image):
    #     image = image.astype(np.float32)[:, :, ::-1]
    #     image = image / 255.0
    #     image -= self.mean
    #     image /= self.std
    #     return image

    # def label_transform(self, label):
    #     return np.array(label).astype('int32')

    # def pad_image(self, image, h, w, size, padvalue):
    #     pad_image = image.copy()
    #     pad_h = max(size[0] - h, 0)
    #     pad_w = max(size[1] - w, 0)
    #     if pad_h > 0 or pad_w > 0:
    #         pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
    #                                        pad_w, cv2.BORDER_CONSTANT,
    #                                        value=padvalue)
    #
    #     return pad_image
    #
    # def rand_crop(self, image, label):
    #     h, w = image.shape[:-1]
    #     image = self.pad_image(image, h, w, self.crop_size,
    #                            (0.0, 0.0, 0.0))
    #     label = self.pad_image(label, h, w, self.crop_size,
    #                            (self.ignore_label,))
    #
    #     new_h, new_w = label.shape
    #     x = random.randint(0, new_w - self.crop_size[1])
    #     y = random.randint(0, new_h - self.crop_size[0])
    #     image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
    #     label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
    #
    #     return image, label

    def multi_scale_aug(self, image, label=None,
                        rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image

        # if rand_crop:
        #     image, label = self.rand_crop(image, label)

        return image, label

    # def resize_short_length(self, image, label=None, short_length=None, fit_stride=None, return_padding=False):
    #     h, w = image.shape[:2]
    #     if h < w:
    #         new_h = short_length
    #         new_w = np.int(w * short_length / h + 0.5)
    #     else:
    #         new_w = short_length
    #         new_h = np.int(h * short_length / w + 0.5)
    #     image = cv2.resize(image, (new_w, new_h),
    #                        interpolation=cv2.INTER_LINEAR)
    #     pad_w, pad_h = 0, 0
    #     if fit_stride is not None:
    #         pad_w = 0 if (new_w % fit_stride == 0) else fit_stride - (new_w % fit_stride)
    #         pad_h = 0 if (new_h % fit_stride == 0) else fit_stride - (new_h % fit_stride)
    #         image = cv2.copyMakeBorder(
    #             image, 0, pad_h, 0, pad_w,
    #             cv2.BORDER_CONSTANT, value=tuple(x * 255 for x in self.mean[::-1])
    #         )
    #
    #     if label is not None:
    #         label = cv2.resize(
    #             label, (new_w, new_h),
    #             interpolation=cv2.INTER_NEAREST)
    #         if pad_h > 0 or pad_w > 0:
    #             label = cv2.copyMakeBorder(
    #                 label, 0, pad_h, 0, pad_w,
    #                 cv2.BORDER_CONSTANT, value=self.ignore_label
    #             )
    #         if return_padding:
    #             return image, label, (pad_h, pad_w)
    #         else:
    #             return image, label
    #     else:
    #         if return_padding:
    #             return image, (pad_h, pad_w)
    #         else:
    #             return image

    # def random_brightness(self, img):
    #     if not config.TRAIN.RANDOM_BRIGHTNESS:
    #         return img
    #     if random.random() < 0.5:
    #         return img
    #     self.shift_value = config.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE
    #     img = img.astype(np.float32)
    #     shift = random.randint(-self.shift_value, self.shift_value)
    #     img[:, :, :] += shift
    #     img = np.around(img)
    #     img = np.clip(img, 0, 255).astype(np.uint8)
    #     return img

    # def gen_sample(self, image, label,
    #                multi_scale=True, is_flip=True):
    #     if multi_scale:
    #         rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
    #         image, label = self.multi_scale_aug(image, label,
    #                                             rand_scale=rand_scale)
    #
    #     image = self.random_brightness(image)
    #     image = self.input_transform(image)
    #     label = self.label_transform(label)
    #
    #     image = image.transpose((2, 0, 1))
    #
    #     if is_flip:
    #         flip = np.random.choice(2) * 2 - 1
    #         image = image[:, :, ::flip]
    #         label = label[:, ::flip]
    #
    #     if self.downsample_rate != 1:
    #         label = cv2.resize(
    #             label,
    #             None,
    #             fx=self.downsample_rate,
    #             fy=self.downsample_rate,
    #             interpolation=cv2.INTER_NEAREST
    #         )
    #
    #     return image, label

    # def reduce_zero_label(self, labelmap):
        labelmap = np.array(labelmap)
        encoded_labelmap = labelmap - 1

        return encoded_labelmap

    def inference(self, config, model, image, flip=False):
        size = image.size()
        pred = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]

        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))

            if config.MODEL.NUM_OUTPUTS > 1:
                flip_output = flip_output[config.TEST.OUTPUT_INDEX]

            flip_output = F.interpolate(
                input=flip_output, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(
                flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()
