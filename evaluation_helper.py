import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def getIoU(mask1,mask2):
    # Calculate intersection
    intersection = np.logical_and(mask1, mask2)

    # plt.imshow(intersection)

    # Calculate union
    union = np.logical_or(mask1, mask2)

    # Compute IoU
    iou = np.sum(intersection) / np.sum(union)

    # print("Intersection:", intersection)
    # print("Union:", union)
    # print("IoU:", iou)
    return iou


def getSAD(rgba_img, alpha_img):
    # Extract the alpha channel
    alpha_channel1 = rgba_img[:, :, 3]
    # Calculate the SAD between the two images
    sad = np.sum(np.abs(alpha_channel1 - alpha_img))

    return sad

def getMSE(rgba_img,alpha_img):
    # Extract the alpha channel
    alpha_channel1 = rgba_img[:, :, 3]

    # Calculate the squared difference
    diff = alpha_channel1 - alpha_img
    squared_diff = diff**2

    # Calculate the mean squared error (MSE)
    mse = np.mean(squared_diff)
    # print("MSE:", mse)
    # mse = mean_squared_error(alpha_channel1.flatten(), alpha_img.flatten())    
    return mse