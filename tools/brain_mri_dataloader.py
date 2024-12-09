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

import cv2
import numpy as np
import random

import torch
from torch.nn import functional as F
from torch.utils import data

from config import config_hrnet_v2 as config


class BMDataLoader(torch.utils.data.Dataset):
    
    def __init__(self, output_image_height=700, images=None,
                 masks=None, normalizer=None, channel_values=None):
        
        self.output_image_height    = output_image_height
        self.images                 = images
        self.masks                  = masks
        self.normalizer             = normalizer
        self.channel_values         = channel_values

        if not self.channel_values:
            self.label_dictionary = {
                0:  {'name': 'background',      'train_id': 0,   'color': (10, 206, 0)},
                1:  {'name': 'foreground',      'train_id': 1,   'color': (3, 0, 177)}
                }
        else:
            self.label_dictionary = self.channel_values

        self.length = len(self.images)
        
        #import pdb
        print(f"BMLoader: total images: {len(self.images)}")
        print(f"BMLoader: total masks: {len(self.masks)}")
        print(f"BMLoader: labels: {self.label_dictionary}")
        print(f"BMLoader: transform: {self.normalizer}")

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
            #import pdb
            #pdb.set_trace()
            image, label_image_gray = self.normalizer(img, label_image_gray)
        else:
            raise NotImplementedError("Normalizer not implemented...")

        return image, label_image_gray

    def __len__(self):
        return len(self.images)
