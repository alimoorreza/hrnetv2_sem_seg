import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from imageio import imread
import PIL.Image
import cv2
import os
from scipy.io import loadmat


class UWFSDataLoader(torch.utils.data.Dataset):
    def __init__(self, output_image_height=700, dataset_root=None,
                 image_format='png', indices=None, channel_values=None,
                 normalizer=None):
        self.output_image_height = output_image_height
        self.dataset_root = dataset_root
        self.image_format = image_format
        self.indices = indices
        self.channel_values = channel_values
        self.normalizer = normalizer
        self.mat_files = []
        self.image_size = (output_image_height, output_image_height)

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

        if not self.dataset_root or not os.path.exists(self.dataset_root):
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(self.dataset_root))
        if not self.indices:
            raise FileNotFoundError('No dataset files found.')

        for index in self.indices:
            self.mat_files.append(os.path.join(self.dataset_root, f'{index}'))
        self.length = len(self.mat_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(self.dataset_root))

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_label_dict(self):
        return self.label_dictionary

    def get_image_nd_label(self, index):
        mat = loadmat(self.mat_files[index])
        image = mat['image_array']
        label = mat['mask_array']
        # print(np.unique(label))
        img_h, img_w = image.shape[0], image.shape[1]
        ratio = img_w / img_h
        out_width = int(ratio * self.output_image_height)
        output_size= (self.output_image_height, self.output_image_height)
        if (image.shape[0], image.shape[1]) != output_size:
            image = cv2.resize(image, output_size, 0, 0, interpolation=cv2.INTER_LINEAR)
        if (label.shape[0], label.shape[1]) != output_size:
            label = cv2.resize(label, output_size, 0, 0, interpolation=cv2.INTER_NEAREST)
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
        return len(self.mat_files)
