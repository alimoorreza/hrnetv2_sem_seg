import random
import os
import numpy as np
import shutil
from tqdm import tqdm


def gen_train_test_indices(train_indices_count=2500, val_indices_count=500, val_test_rat=1,
                           start_index=1, end_index=24966, data_set_root=None,
                           out_dataset_root=None, subset_folders=False):
    if not data_set_root or not os.path.exists(data_set_root):
        raise FileNotFoundError('No dataset files found.')

    if not out_dataset_root:
        raise FileNotFoundError('Enter Valid Output Folder.')

    n = train_indices_count + val_indices_count + (val_indices_count//val_test_rat)

    m = val_indices_count + (val_indices_count // val_test_rat)

    train_val_indices = random.sample(range(start_index, end_index), n)
    # print(len(np.unique(np.array(train_val_indices))))

    val_indices_selector = random.sample(range(0, len(train_val_indices)-1), m)
    # print(len(np.unique(np.array(val_indices_selector))))

    train_indices = []
    val_indices = []
    test_indices = []
    d_rat = 0
    flg = True
    if subset_folders:
        if not os.path.exists(os.path.join(out_dataset_root, 'images')):
            os.makedirs(os.path.join(out_dataset_root, 'images'))
        if not os.path.exists(os.path.join(out_dataset_root, 'labels')):
            os.makedirs(os.path.join(out_dataset_root, 'labels'))

    for ind in tqdm(range(len(train_val_indices))):
        if ind not in val_indices_selector:
            train_indices.append(str(train_val_indices[ind]).zfill(5))
            if subset_folders:
                src = os.path.join(data_set_root, 'images', f'{str(train_val_indices[ind]).zfill(5)}.png')
                dst = os.path.join(out_dataset_root, 'images', f'{str(train_val_indices[ind]).zfill(5)}.png')
                shutil.copyfile(src, dst)
                src = os.path.join(data_set_root, 'labels', f'{str(train_val_indices[ind]).zfill(5)}.png')
                dst = os.path.join(out_dataset_root, 'labels', f'{str(train_val_indices[ind]).zfill(5)}.png')
                shutil.copyfile(src, dst)
        else:
            if flg:
                val_indices.append(str(train_val_indices[ind]).zfill(5))
                d_rat += 1
                if d_rat == val_test_rat:
                    d_rat = 0
                    flg = False
            else:
                test_indices.append(str(train_val_indices[ind]).zfill(5))
                flg = True
            if subset_folders:
                src = os.path.join(data_set_root, 'images', f'{str(train_val_indices[ind]).zfill(5)}.png')
                dst = os.path.join(out_dataset_root, 'images', f'{str(train_val_indices[ind]).zfill(5)}.png')
                shutil.copyfile(src, dst)
                src = os.path.join(data_set_root, 'labels', f'{str(train_val_indices[ind]).zfill(5)}.png')
                dst = os.path.join(out_dataset_root, 'labels', f'{str(train_val_indices[ind]).zfill(5)}.png')
                shutil.copyfile(src, dst)

    # list1_as_set = set(train_indices)
    # intersection = list1_as_set.intersection(val_indices)
    # intersection_as_list = list(intersection)
    # print(intersection_as_list, train_indices, val_indices)

    if subset_folders:
        train_indices_file = os.path.join(out_dataset_root, 'train_indices.txt')
        val_indices_file = os.path.join(out_dataset_root, 'val_indices.txt')
        test_indices_file = os.path.join(out_dataset_root, 'test_indices.txt')
    else:
        train_indices_file = os.path.join(data_set_root, 'train_indices.txt')
        val_indices_file = os.path.join(data_set_root, 'val_indices.txt')
        test_indices_file = os.path.join(data_set_root, 'test_indices.txt')

    with open(train_indices_file, 'w') as f:
        f.write(','.join(train_indices))

    with open(val_indices_file, 'w') as f:
        f.write(','.join(val_indices))

    with open(test_indices_file, 'w') as f:
        f.write(','.join(test_indices))


if __name__ == '__main__':
    root = '/Users/imrankabir/Desktop/semantic_seg_audio_description/'
    dataset_root = os.path.join(root, 'game_dataset')
    output_dataset_root = os.path.join(root, 'game_dataset_subset')
    crate_subset_folders = False
    gen_train_test_indices(train_indices_count=2500, val_indices_count=500, val_test_rat=1,
                           start_index=1, end_index=24966,
                           data_set_root=dataset_root,
                           out_dataset_root=output_dataset_root,
                           subset_folders=crate_subset_folders)
