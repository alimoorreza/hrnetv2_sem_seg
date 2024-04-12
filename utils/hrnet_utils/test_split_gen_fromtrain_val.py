import random
import os
import numpy as np
import shutil
from tqdm import tqdm


def main_f(start_index, end_index,
           data_set_root, train_ind_file,
           val_ind_file, val_tst_rat,
           subset_folders, out_dataset_root):
    with open(train_ind_file, 'r') as f:
        train_ind = f.read().strip().replace('\n', '').split(',')

    with open(val_ind_file, 'r') as f:
        val_ind = f.read().strip().replace('\n', '').split(',')

    all_i = list(np.arange(start_index, end_index+1))
    # print(len(list(np.unique(np.array(all_i)))))
    for vl in train_ind:
        k = all_i.index(int(vl))
        del all_i[k]

    for vl in val_ind:
        k = all_i.index(int(vl))
        del all_i[k]

    # print(len(all_i))
    # return

    test_indices_selector = random.sample(range(0, len(all_i) - 1), len(val_ind)//val_tst_rat)

    if subset_folders:
        if not os.path.exists(os.path.join(out_dataset_root, 'images')):
            os.makedirs(os.path.join(out_dataset_root, 'images'))
        if not os.path.exists(os.path.join(out_dataset_root, 'labels')):
            os.makedirs(os.path.join(out_dataset_root, 'labels'))

    test_ind = []
    for ind in tqdm(test_indices_selector):
        test_ind.append(str(all_i[ind]).zfill(5))
        if subset_folders:
            src = os.path.join(data_set_root, 'images', f'{str(all_i[ind]).zfill(5)}.png')
            dst = os.path.join(out_dataset_root, 'images', f'{str(all_i[ind]).zfill(5)}.png')
            shutil.copyfile(src, dst)
            src = os.path.join(data_set_root, 'labels', f'{str(all_i[ind]).zfill(5)}.png')
            dst = os.path.join(out_dataset_root, 'labels', f'{str(all_i[ind]).zfill(5)}.png')
            shutil.copyfile(src, dst)

    if subset_folders:
        test_indices_file = os.path.join(out_dataset_root, 'test_indices.txt')
    else:
        test_indices_file = os.path.join(data_set_root, 'test_indices.txt')

    with open(test_indices_file, 'w') as f:
        f.write(','.join(test_ind))


if __name__ == '__main__':
    root = '/Users/imrankabir/Desktop/semantic_seg_audio_description/'
    dataset_root = os.path.join(root, 'game_dataset')
    output_dataset_root = os.path.join(root, 'game_dataset_subset_test_only')
    tr_f = '/Users/imrankabir/Desktop/semantic_seg_audio_description/hrnet_semantic_segmentation_on_GTA/hrnet_semantic_training/experiments/tr.txt'
    vl_f = '/Users/imrankabir/Desktop/semantic_seg_audio_description/hrnet_semantic_segmentation_on_GTA/hrnet_semantic_training/experiments/val.txt'
    crate_subset_folders = True
    main_f(start_index=1, end_index=24966,
           data_set_root=dataset_root,
           train_ind_file=tr_f, val_ind_file=vl_f,
           val_tst_rat=1, subset_folders=crate_subset_folders,
           out_dataset_root=output_dataset_root)
