import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

dataset = 'greendino_new'
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
        m = cv2.imread(mask_path)
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        m = cv2.resize(m, (h, w))

        if invert:
            _, m = cv2.threshold(m, 255//2, 255, cv2.THRESH_BINARY_INV)
        else:
            _, m = cv2.threshold(m, 255//2, 255, cv2.THRESH_BINARY)
        print(np.sum(m == 0) + np.sum(m == 255), m.size)
        name, ext = mask.split('.')
        print(os.path.join(mask_folder, name + '.jpg'))
        cv2.imwrite(os.path.join(mask_folder, name + '.jpg'), m)

    
