import numpy as np
from scipy.io import loadmat,savemat
import os
import joblib
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,IterableDataset
from torchvision.transforms import Compose
import random
from PIL import Image
from glob import glob
import torchvision.transforms.functional as tr_F
from scipy import ndimage
import itertools
import time
import random
import math
import numpy as np
import numbers
import collections
import cv2

import torch

# manual_seed = 123
# torch.manual_seed(manual_seed)
# np.random.seed(manual_seed)
# torch.manual_seed(manual_seed)
# torch.cuda.manual_seed_all(manual_seed)
# random.seed(manual_seed)


class RandomMirror(object):
    """
    Randomly filp the images/masks horizontally
    """
    def __call__(self, sample):
        support, support_fg, support_bg = sample['support_image'], sample['support_fg_mask'],sample['support_bg_mask']
        if random.random() < 0.5:
            support = [i.transpose(Image.FLIP_LEFT_RIGHT) for i in support]
            support_fg = [i.transpose(Image.FLIP_LEFT_RIGHT) for i in support_fg]
            support_bg = [i.transpose(Image.FLIP_LEFT_RIGHT) for i in support_bg]

        sample['support_image'] = support
        sample['support_fg_mask'] = support_fg
        sample["support_bg_mask"] = support_bg
        return sample


class Resize(object):
    """
    Resize images/masks to given size
    Args:
        size: output size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        support, support_fg, support_bg = sample['support_image'], sample['support_fg_mask'],sample['support_bg_mask']
        query = sample["query_image"]
        query_label = sample["query_label"]
        support = [tr_F.resize(i, self.size) for i in support]
        query = [tr_F.resize(i, self.size) for i in query]
        support_fg = [tr_F.resize(i, self.size, interpolation=Image.NEAREST) for i in support_fg]
        support_bg = [tr_F.resize(i, self.size, interpolation=Image.NEAREST) for i in support_bg]
        query_label = [tr_F.resize(i, self.size, interpolation=Image.NEAREST) for i in query_label]

        """support = [Image.fromarray(resize(np.array(i), self.size).astype(np.uint8)) for i in support]
        query = [Image.fromarray(resize(np.array(i), self.size).astype(np.uint8)) for i in query]
        support_fg = [Image.fromarray(resize(np.array(i), self.size, order=0, preserve_range=True).astype(np.uint8)) for i in support_fg]
        support_bg = [Image.fromarray(resize(np.array(i), self.size, order=0, preserve_range=True).astype(np.uint8)) for i in support_bg]
        query_label = [Image.fromarray(resize(np.array(i), self.size, order=0, preserve_range=True).astype(np.uint8)) for i in query_label]"""

        sample['support_image'] = support
        sample['query_image'] = query
        sample['support_fg_mask'] = support_fg
        sample["support_bg_mask"] = support_bg
        sample["query_label"] = query_label
        return sample


class ToTensorNormalize(object):
    """
    Convert images/masks to torch.Tensor
    Scale images' pixel values to [0-1] and normalize with predefined statistics
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        support, support_fg, support_bg = sample['support_image'], sample['support_fg_mask'],sample['support_bg_mask']
        query = sample["query_image"]
        query_label = sample["query_label"]
        support = [tr_F.to_tensor(i) for i in support]
        query = [tr_F.to_tensor(i) for i in query]
        support = [tr_F.normalize(i, mean=self.mean, std=self.std) for i in support]
        query = [tr_F.normalize(i, mean=self.mean, std=self.std) for i in query]

        sample['support_image'] = [i for i in support]
        sample["query_image"] = [i for i in query]

        sample['support_fg_mask'] = [torch.Tensor(np.array(i)).long().float() for i in support_fg]
        sample["support_bg_mask"] = [torch.Tensor(np.array(i)).long().float() for i in support_bg]
        sample["query_label"] = [torch.Tensor(np.array(i)).long() for i in query_label]
        return sample


"""class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, sample):
        support, support_fg, support_bg = sample['support_image'], sample['support_fg_mask'], sample['support_bg_mask']
        query = sample["query_image"]
        query_label = sample["query_label"]

        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        support = [Image.fromarray(cv2.resize(
            np.array(i),
            None,
            fx=scale_factor_x,
            fy=scale_factor_y,
            interpolation=cv2.INTER_LINEAR)) for i in support]
        query = [Image.fromarray(cv2.resize(
            np.array(i),
            None,
            fx=scale_factor_x,
            fy=scale_factor_y,
            interpolation=cv2.INTER_LINEAR)) for i in query]
        support_fg = [Image.fromarray(cv2.resize(
            np.array(i),
            None,
            fx=scale_factor_x,
            fy=scale_factor_y,
            interpolation=cv2.INTER_NEAREST)) for i in support_fg]
        support_bg = [Image.fromarray(cv2.resize(
            np.array(i),
            None,
            fx=scale_factor_x,
            fy=scale_factor_y,
            interpolation=cv2.INTER_NEAREST)) for i in support_bg]
        query_label = [Image.fromarray(cv2.resize(
            np.array(i),
            None,
            fx=scale_factor_x,
            fy=scale_factor_y,
            interpolation=cv2.INTER_NEAREST)) for i in query_label]

        sample['support_image'] = support
        sample['query_image'] = query
        sample['support_fg_mask'] = support_fg
        sample["support_bg_mask"] = support_bg
        sample["query_label"] = query_label

        return sample"""
