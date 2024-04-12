import cv2
import numpy as np
import glob
import os

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('001.avi', fourcc, 1, (1024, 520))

root = '/Users/imrankabir/Desktop/semantic_seg_audio_description/'

img_folder = os.path.join(root, 'out_images_real')

output_vid = os.path.join(root, 'test_out.mp4')

for filename in glob.glob(os.path.join(img_folder, '*.png')):
    img = cv2.imread(filename)
    video.write(img)

cv2.destroyAllWindows()
video.release()

"""root = '/Users/imrankabir/Desktop/semantic_seg_audio_description/'

img_folder = os.path.join(root, 'out_images')

output_vid = os.path.join(root, 'test_out.mp4')

size = (1025, 1025)
fps = 25
img_array = []
for filename in glob.glob(os.path.join(img_folder, '*.png')):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(output_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()
cv2.destroyAllWindows()"""
