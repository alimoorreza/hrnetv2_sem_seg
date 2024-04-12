import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dice_loss_func(input_, target_):
    """
    input_ is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input_
    """
    target_ = target_.cuda()
    target = torch.zeros(input_.size())
    target = target.cuda()
    target.scatter_(1, target_.view(target_.size(0), 1, target_.size(1), target_.size(2)), 1)

    # print(input_.size(),target.size(),input_.shape,target.shape)
    assert input_.size() == target.size(), "input_ sizes must be equal."
    assert input_.dim() == 4, "input_ must be a 4D Tensor."
    #     uniques=np.unique(target.numpy())
    #     assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    prob_s = F.softmax(input_)
    num = prob_s * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)  # b,c,h
    num = torch.sum(num, dim=2)

    den1 = prob_s * prob_s  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2 * (num / (den1 + den2))
    dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg

    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

    return dice_total


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, input_s, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        input_s = F.sigmoid(input_s)

        # flatten label and prediction tensors
        input_s = input_s.view(-1)
        targets = targets.view(-1)

        intersection = (input_s * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (input_s.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(input_s, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE