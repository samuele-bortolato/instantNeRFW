import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

dataset = 'yellowdino_new_new'
invert = True

datasets_folder = os.path.join(os.getcwd(), 'datasets')
mask_folder = os.path.join(datasets_folder, dataset, 'masks')
image_folder = os.path.join(datasets_folder, dataset, 'images')

image = os.path.join(image_folder, os.listdir(image_folder)[0])
h = cv2.imread(image).shape[0]
w = cv2.imread(image).shape[1]

for mask in os.listdir(mask_folder):
    name, ext = mask.split('.')
    if ext == 'png':
        mask_path = os.path.join(mask_folder, mask)
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (h, w))
        if invert:
            _, m = cv2.threshold(m, 255//2, 255, cv2.THRESH_BINARY_INV)
        else:
            _, m = cv2.threshold(m, 255//2, 255, cv2.THRESH_BINARY)
        if np.sum(m == 0) + np.sum(m == 255)!= m.size:
            raise "Mask are not binary!!"
        name, ext = mask.split('.')
        print(mask_path)
        cv2.imwrite(mask_path, m)