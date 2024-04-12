from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.DEVICE = 'cuda'
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'seg_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.NUM_CLASSES = 30
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [1050, 1050]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False
_C.LOSS.IGNORE_LABEL = 255

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'mpii'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.VAL_SET = 'valid'
_C.DATASET.TEST_SET = 'test'
_C.DATASET.DATA_FORMAT = 'png'
_C.DATASET.DATA_FORMAT_LABEL = 'png'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False
_C.DATASET.IMAGE_W = 1914
_C.DATASET.IMAGE_H = 1052
_C.DATASET.USE_MGDA = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

_C.TRAIN.ARCH = 'hrnet'
_C.TRAIN.TRAIN_H = 1050
_C.TRAIN.TRAIN_W = 1050
_C.TRAIN.SCALE_MIN = 0.5
_C.TRAIN.SCALE_MAX = 1.0
_C.TRAIN.SHORT_SIZE = 1080
_C.TRAIN.ROTATE_MIN = -10
_C.TRAIN.ROTATE_MAX = 10
_C.TRAIN.ZOOM_FACTOR = 8
_C.TRAIN.IGNOE_LABEL = 255
_C.TRAIN.AUX_WEIGHT = 0.4
_C.TRAIN.NUM_EXAMPLES = 1000000
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.BATCH_SIZE_VAL = 1
_C.TRAIN.BASE_LR = 0.01
_C.TRAIN.END_EPOCH = 100
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.POWER = 0.9
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0001
_C.TRAIN.PRINT_FREQ = 100
_C.TRAIN.SAVE_FREQ = 1
_C.TRAIN.SAVE_PATH = ''
_C.TRAIN.RESUME = True
_C.TRAIN.AUTO_RESUME = None
_C.TRAIN.EVALUATE = True
_C.TRAIN.PRETRAINED_MODEL = ''

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False

_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''
_C.TEST.IOU = 0.8
_C.TEST.IMAGE_CONVERSION_TH = 0.5
_C.TEST.VIDEO_PATH = ''


# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.config)

    """if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    if args.testModel:
        cfg.TEST.MODEL_FILE = args.testModel"""

    # cfg.DATASET.ROOT = os.path.join(
    #     cfg.DATA_DIR, cfg.DATASET.DATASET, 'images')

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

