import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import PIL.Image
import os
import cv2


def alpha_blender(fore_ground, back_ground, alpha):
    fore_ground = np.multiply(fore_ground, alpha)
    back_ground = np.multiply(back_ground, 1-alpha)
    out_img = np.add(fore_ground, back_ground).astype(np.uint8)
    return out_img


def alpha_blender_cv2(fore_ground, back_ground, alpha):
    foreground = fore_ground.astype(float)
    background = back_ground.astype(float)
    alpha = alpha.astype(float) / 255
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    outImage = cv2.add(foreground, background)
    return outImage


if __name__ == '__main__':
    root = '/Users/imrankabir/Desktop/semantic_seg_audio_description/'
    dataset_root = os.path.join(root, 'game_dataset')
    image = imread(
        os.path.join(dataset_root, 'images', '00100.png')
    )
    label_image = PIL.Image.open(
        os.path.join(dataset_root, 'labels', '00100.png')
    )
    label_image = label_image.convert('RGB')
    label_image = np.array(label_image)
    out_ = alpha_blender(fore_ground=image, back_ground=label_image, alpha=0.5)
    plt.imshow(out_)
    plt.show()

