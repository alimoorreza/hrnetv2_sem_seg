import glob
import random
import os

root = '/Users/imrankabir/Desktop/research/semantic_seg_audio_description/dataset_gta_2/'

train_root = os.path.join(root, 'test/')

images = glob.glob(os.path.join(train_root, 'img/*/*.jpg'))

train_indices = random.sample(range(0, len(images)-1), 500)
print(len(images))
final_images = []

for ind in train_indices:
    td = images[ind].replace(train_root, '').replace('.jpg', '').replace('img/', '')
    final_images.append(td)

print(len(final_images))

with open('test_indices.txt', 'w') as f:
    f.write(','.join(final_images))


