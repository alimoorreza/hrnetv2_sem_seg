# MAKE_DATASET_DIR = "/N/u/sshubham/Carbonate/few_shot_learning/new_mat_files/"
# SAVE_DATASET_DIR = "/N/slate/sshubham/few_shots_segmentation/dataset5/"
import sys
import torch

BATCH_SIZE = 1
IMAGE_RESIZE = (417, 417)
NUM_EPOCH = 40

try:
    TEST_LABEL_SPLIT_VALUE = int(sys.argv[1])
except:
    TEST_LABEL_SPLIT_VALUE = 0

print("TEST_LABEL_SPLIT_VALUE", TEST_LABEL_SPLIT_VALUE)

NWAYS = 1
NSHOTS = int(sys.argv[2])

VGG_MODEL_PATH = "./vgg16-397923af.pth"
MODEL_SAVE_DIR = "../model/"


USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print('USE GPU')
else:
    print('USE CPU')
