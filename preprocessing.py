# -*- coding:utf-8 -*-
import os
import sys
from config_training import config
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.morphology import disk


def load_image():
    file_list = os.path.join(config['training_data_path'], 'train.txt')
    image_paths = []

    with open(file_list, 'r') as f:
        for line in f:
            line = line.strip().split()[0]
            image_paths.append(line)
    image_paths = [x[39:] for x in image_paths]
    image_paths = [os.path.join(config['training_data_path'], x)
                   for x in image_paths]

    training_set = [io.imread(x) for x in image_paths]  # 952 * (256,256,3)
    return training_set


def mask(training_set):
    for img in training_set:
        img = color.rgb2hsv(img)
        threshold_h = threshold_otsu(img[:,:,0])
        threshold_s = threshold_otsu(img[:,:,1])
        mask_h = img[:,:,1] > threshold_h
        mask_s = img[:,:,1] > threshold_s
        # img[:,:,0] = mask_h.astype(np.uint8)
        # img[:,:,1] = mask_s.astype(np.uint8)

        mask = (mask_h & mask_s).astype(np.uint8)
        img[:,:,0] = mask
        img[:,:,1] = mask
        img[:,:,2] = mask
    return training_set


def main():
    training_set = load_image()
    # plt.imshow(training_set[0])
    # plt.show()
    training_set = mask(training_set)

    # plt.imshow(training_set[0])
    # plt.show()

if __name__ == '__main__':
    main()
