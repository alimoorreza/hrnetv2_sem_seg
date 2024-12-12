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


class PartSegDataLoader(torch.utils.data.Dataset):
    def __init__(self, output_image_height=700, images=None,
                 masks=None, normalizer=None, channel_values=None):
        self.output_image_height = output_image_height
        self.images = images
        self.masks = masks
        self.normalizer = normalizer
        self.channel_values = channel_values

        if not self.channel_values:
            self.label_dictionary = {
                0:  {'name': 'background',  'train_id': 0,     'color': (100, 50,   50)},
                1:  {'name': 'head',        'train_id': 1,     'color': (128, 64,  128)},
                2:  {'name': 'body',        'train_id': 2,     'color': (244, 35,  232)},
                3:  {'name': 'fin',         'train_id': 3,     'color': (70,  70,  70)},
                4:  {'name': 'tail',        'train_id': 4,     'color': (102, 102, 156)},
                
            }
        else:
            self.label_dictionary = self.channel_values

        self.length = len(self.images)

        print("the class labeling is: ", self.label_dictionary)
                        
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
        return image, label

    def __getitem__(self, index):
        img, label_image_gray = self.get_image_nd_label(index)
                
        if self.normalizer:            
            image, label_image_gray = self.normalizer(img, label_image_gray)
            #print(f"shape of img: {image.shape}")
            #print(f"shape of label: {label_image_gray.shape}")
                 
            
        else:
            raise NotImplementedError("Normalizer not implemented...")            

        return image, label_image_gray

    def __len__(self):
        return len(self.images)
