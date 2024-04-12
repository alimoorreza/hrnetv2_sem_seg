import numpy as np
from scipy.io import loadmat, savemat
import io
import os
from glob import glob
import joblib
from tqdm import tqdm

data_files = os.listdir("/Users/imrankabir/Desktop/few_shot/new_mat_files/")

classes2labels = {
    'crab_sem_seg_gt': 1,
    'crocodile_sem_seg_gt': 2,
    'dolphin_sem_seg_gt': 3,
    'frog_sem_seg_gt': 4,
    'nettles_sem_seg_gt': 5,
    'octopus_sem_seg_gt': 6,
    'otter_sem_seg_gt': 7,
    'penguin_sem_seg_gt': 8,
    'polar_bear_sem_seg_gt': 9,
    'sea_anemone_sem_seg_gt': 10,
    'sea_urchin_sem_seg_gt': 11,
    'seahorse_sem_seg_gt': 12,
    'seal_sem_seg_gt': 13,
    'shark_sem_seg_gt': 14,
    'shrimp_sem_seg_gt': 15,
    'star_fish_sem_seg_gt': 16,
    'stingray_sem_seg_gt': 17,
    'squid_sem_seg_gt': 18,
    'turtle_sem_seg_gt': 19,
    'whale_sem_seg_gt': 20,
    'nudibranch_sem_seg_gt': 21,
}
print(classes2labels)
classes = [x.replace('.mat', '') for x in data_files]
for class_ in tqdm(classes):
    if not os.path.exists(f"../dataset/{class_}"):
        os.makedirs(f"../dataset/{class_}")
    data = loadmat(f"/Users/imrankabir/Desktop/few_shot/new_mat_files/{class_}.mat")
    for sample in tqdm(data['Image2GTMapping'][0]):
        filename = sample[0][0]
        dict_ = {}
        dict_["class"] = class_
        dict_["image_array"] = sample[1]
        dict_["mask_array"] = sample[2]
        savemat(f"../dataset/{class_}/{filename}.mat", dict_)

joblib.dump(classes2labels, "../model/classes2labels.joblib")


classes2labels = {
    'unlabeled': 255,
    'crab': 1,
    'crocodile': 2,
    'dolphin': 3,
    'frog': 4,
    'nettles': 5,
    'octopus': 6,
    'otter': 7,
    'penguin': 8,
    'polar_bear': 9,
    'sea_anemone': 10,
    'sea_urchin': 11,
    'seahorse': 12,
    'seal': 13,
    'shark': 14,
    'shrimp': 15,
    'star_fish': 16,
    'stingray': 17,
    'squid': 18,
    'turtle': 19,
    'whale': 20,
    'nudibranch': 21,
    'coral': 22,
    'rock': 23,
    'water': 24,
    'sand': 25,
    'plant': 26,
    'human': 27,
    'iceberg': 28,
    'reef': 29,
    'dynamic': 30
}
