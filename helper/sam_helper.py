import random
import numpy as np
import matplotlib.pyplot as plt
from helper import *
# import torch
# import math

# import cv2
# from PIL import Image
# from scipy.ndimage import binary_dilation, binary_erosion


def random_color():
    # Generate random values for red, green, and blue
    r, g, b = [random.randint(0, 255) for i in range(3)]
    # Create a numpy array with the random RGB values and fixed alpha value of 255
    color = np.array([r, g, b, 255], dtype=np.uint8)
    return color

# print(color)  # prints something like [112  43 194 255]


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


