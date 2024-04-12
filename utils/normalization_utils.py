#!/usr/bin/python3

import numpy as np
import torch

from typing import Optional, Tuple


def get_imagenet_mean_std(scale_val):
    value_scale = scale_val
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    return mean, std
            

def normalize_img(input_: torch.Tensor,
                  mean: Tuple[float, float, float],
                  std: Optional[Tuple[float, float, float]] = None):

    if std is None:
        for t, m in zip(input_, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input_, mean, std):
            t.sub_(m).div_(s)

