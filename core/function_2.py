import _init_paths

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.hrnet_v2_utils.utils import AverageMeter
from utils.hrnet_v2_utils.utils import get_confusion_matrix
from utils.hrnet_v2_utils.utils import adjust_learning_rate

import utils.hrnet_v2_utils.distributed as dist
from PIL import Image


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        losses, _ = model(images, labels)
        loss = losses.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    img_save_counter = 0
    img_folder = 'results/images'
    mask_folder = 'results/masks'
    mask_folder_3 = 'results/masks_3'
    root_mask_folder_3 = 'results/root_masks_3'
    label_dictionary = {
        0: {'name': 'unlabeled', 'train_id': 0, 'color': (0, 0, 0)},
        1: {'name': 'crab', 'train_id': 1, 'color': (128, 64, 128)},
        2: {'name': 'crocodile', 'train_id': 2, 'color': (244, 35, 232)},
        3: {'name': 'dolphin', 'train_id': 3, 'color': (70, 70, 70)},
        4: {'name': 'frog', 'train_id': 4, 'color': (102, 102, 156)},
        5: {'name': 'nettles', 'train_id': 5, 'color': (190, 153, 153)},
        6: {'name': 'octopus', 'train_id': 6, 'color': (153, 153, 153)},
        7: {'name': 'otter', 'train_id': 7, 'color': (250, 170, 30)},
        8: {'name': 'penguin', 'train_id': 8, 'color': (220, 220, 0)},
        9: {'name': 'polar_bear', 'train_id': 9, 'color': (107, 142, 35)},
        10: {'name': 'sea_anemone', 'train_id': 10, 'color': (152, 251, 152)},
        11: {'name': 'sea_urchin', 'train_id': 11, 'color': (70, 130, 180)},
        12: {'name': 'seahorse', 'train_id': 12, 'color': (220, 20, 60)},
        13: {'name': 'seal', 'train_id': 13, 'color': (255, 0, 0)},
        14: {'name': 'shark', 'train_id': 14, 'color': (0, 0, 142)},
        15: {'name': 'shrimp', 'train_id': 15, 'color': (0, 0, 70)},
        16: {'name': 'star_fish', 'train_id': 16, 'color': (0, 60, 100)},
        17: {'name': 'stingray', 'train_id': 17, 'color': (0, 80, 100)},
        18: {'name': 'squid', 'train_id': 18, 'color': (0, 0, 230)},
        19: {'name': 'turtle', 'train_id': 19, 'color': (119, 11, 32)},
        20: {'name': 'whale', 'train_id': 20, 'color': (111, 74, 0)},
        21: {'name': 'nudibranch', 'train_id': 21, 'color': (81, 0, 81)},
        22: {'name': 'coral', 'train_id': 22, 'color': (250, 170, 160)},
        23: {'name': 'rock', 'train_id': 23, 'color': (230, 150, 140)},
        24: {'name': 'water', 'train_id': 24, 'color': (180, 165, 180)},
        25: {'name': 'sand', 'train_id': 25, 'color': (150, 100, 100)},
        26: {'name': 'plant', 'train_id': 26, 'color': (150, 120, 90)},
        27: {'name': 'human', 'train_id': 27, 'color': (153, 153, 153)},
        28: {'name': 'iceberg', 'train_id': 28, 'color': (0, 0, 90)},
        29: {'name': 'reef', 'train_id': 29, 'color': (0, 0, 110)},
        30: {'name': 'dynamic', 'train_id': 30, 'color': (200, 60, 110)},
        31: {'name': 'others', 'train_id': 31, 'color': (47, 220, 70)}
    }
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    if not os.path.exists(mask_folder_3):
        os.makedirs(mask_folder_3)
    if not os.path.exists(root_mask_folder_3):
        os.makedirs(root_mask_folder_3)
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, root_img, root_mask = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            losses, pred = model(image, label)

            pred_mask = F.interpolate(
                input=pred, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            ).max(1)[1]

            for ind, ri in enumerate(root_img):
                out_img = ri.cpu().numpy()
                out_mask = pred_mask[ind].cpu().numpy()
                out_root_mask = root_mask[ind].cpu().numpy()

                out_mask_3 = np.zeros((out_mask.shape[0], out_mask.shape[1], 3), dtype=np.uint8)
                out_root_mask_3 = np.zeros((out_mask.shape[0], out_mask.shape[1], 3), dtype=np.uint8)

                for key in label_dictionary.keys():
                    out_mask_3[out_mask == label_dictionary[key]['train_id']] = label_dictionary[key]['color']
                    out_root_mask_3[out_root_mask == label_dictionary[key]['train_id']] = label_dictionary[key]['color']

                # print(out_img.shape, out_mask.shape)

                out_img = Image.fromarray(out_img.astype(np.uint8))
                out_mask = Image.fromarray(out_mask.astype(np.uint8), 'L')
                out_mask_3 = Image.fromarray(out_mask_3.astype(np.uint8))
                out_root_mask_3 = Image.fromarray(out_root_mask_3.astype(np.uint8))

                out_img.save(os.path.join(img_folder, f"{img_save_counter}.png"))
                out_mask.save(os.path.join(mask_folder, f"{img_save_counter}.png"))
                out_mask_3.save(os.path.join(mask_folder_3, f"{img_save_counter}.png"))
                out_root_mask_3.save(os.path.join(root_mask_folder_3, f"{img_save_counter}.png"))
                img_save_counter += 1

            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )
            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
