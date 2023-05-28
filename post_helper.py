import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from helper import *

import cv2
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from pymatting import *
from sklearn.metrics import mean_squared_error

# DIS
import sys
sys.path.append('/Users/susanto/Documents/Proyek/best-remove-background/latihan-remove-background/PytorchSuccessExample/algorithm/DIS/IS-Net')
from DIS import getDISMask

# InSPyReNet
from transparent_background import Remover

# Matte former
sys.path.append('/Users/susanto/Documents/Proyek/best-remove-background/latihan-remove-background/matteformer')
import networks
import utils_matteformer as utils
from inference import generator_tensor_dict, single_inference

# AIM NET
sys.path.append('/Users/susanto/Documents/Proyek/best-remove-background/latihan-remove-background/AIM')
sys.path.append('/Users/susanto/Documents/Proyek/best-remove-background/latihan-remove-background/AIM/core')
from aim import *


def get_edge_mask(mask):
    # Dilate the mask to get the boundary pixels
    dilated_mask = binary_dilation(mask)
    # Erode the mask to get the interior pixels
    eroded_mask = binary_erosion(mask)
    # Subtract the eroded mask from the dilated mask to get the boundary pixels
    edge_mask = np.logical_and(dilated_mask, np.logical_not(eroded_mask))
    return edge_mask


# InSPyReNet Remover Tool
model_path_InSPyReNet = "/Users/susanto/Documents/Proyek/best-remove-background/models/InSPyReNet_SwinB_Large.pth"

remover = Remover(fast=True, jit=True, device='cpu',
                  ckpt=model_path_InSPyReNet)  # custom setting




def InSPyReNetRemover(PIL_image):
    return remover.process(PIL_image)


def saveMask(mask, name="example.jpg"):
    # Convert the binary mask to grayscale image
    gray_image = mask.astype(np.uint8) * 255

    a = Image.fromarray(gray_image)
    a.save("output/"+name)


def makeTrimap(img, s_size=2, name="example"):
    # Step 1: Select Only Main Visible Area
    main_area_mask = img[:, :, 3] > 170
    # plt.imshow(main_area_mask)

    # Step 2: Detect Edges (For nets rope, etc)
    # limit edge detection to visible area only
    visible_area = img.copy()
    visible_area[img[:, :, 3] < 200] = [0, 0, 0, 0]
    PIL_image, bounding = cropUnusedBlankPixel(visible_area)
    x, y, x1, y1 = bounding
    croped_img = img[y:y1, x:x1]

    # Convert RGBA image to grayscale
    gray_image = cv2.cvtColor(
        croped_img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    threshold = 0  # Set the threshold value
    edges_mask = edges > threshold

    # Back to original size
    blank_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    blank_img = blank_img != 0
    edges_mask = patchToBoundingBox(blank_img, bounding, edges_mask)

    # Remove outside visible_area
    visible_area = img[:, :, 3] > 20
    edges_mask = np.logical_and(edges_mask, visible_area)
    # plt.imshow(edges_mask)

    # Step 3: Perform Closing Into Edge region
    dilated = binary_dilation(edges_mask, structure=np.ones((s_size, s_size)))
    closed_edge_mask = binary_erosion(
        dilated, structure=np.ones((s_size, s_size)))

    # plt.imshow(dilated)

    # Step 4: main_area_mask + closed_edge_mask
    fore_mask = np.logical_or(main_area_mask, closed_edge_mask)

    # Step 5: Get Transition Mask
    erosion_fore_mask = binary_erosion(
        fore_mask, structure=np.ones((s_size, s_size)))
    dilated_fore_mask = binary_dilation(
        fore_mask, structure=np.ones((s_size, s_size)))
    transition_mask = np.logical_and(
        dilated_fore_mask, np.logical_not(erosion_fore_mask))
    
    # plt.imshow(fore_mask)

    # Add more transition_mask based on closed_edge_mask
    transition_mask = np.logical_or(transition_mask, closed_edge_mask)

    backbone = binary_erosion(
        closed_edge_mask, structure=np.ones((s_size+1, s_size+1)))
    # plt.imshow(backbone)

    absolute_foreground = img[:, :, 3] == 255

    backbone = np.logical_or(absolute_foreground, backbone)

    f = (img[:, :, 3] < 250) & (img[:, :, 3] > 50)

    # Create the trimap
    trimap = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    trimap[fore_mask] = 255
    trimap[transition_mask] = 127
    trimap[f] = 127
    trimap[backbone] = 255


    # saveMask(fore_mask, "fore.jpg")
    # saveMask(transition_mask, "transition_mask.jpg")
    # saveMask(backbone, "backbone.jpg")
    # saveMask(f, "fa.jpg")

    return trimap.astype(np.uint8)


def image_matting_matte_former(image_path, trimap_path):
    # build model
    model = networks.get_generator(is_train=False)
    model.cpu()

    checkpoint_path = "/Users/susanto/Documents/Proyek/best-remove-background/models/matteformer_image_matting.pth"

    # build model
    model = networks.get_generator(is_train=False)
    # model.cpu()

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model.load_state_dict(utils.remove_prefix_state_dict(
        checkpoint['state_dict']), strict=True)

    # inference
    model = model.eval()

    # assume image and mask have the same file name
    image_dict = generator_tensor_dict(image_path, trimap_path)
    alpha_pred = single_inference(model, image_dict)

    return alpha_pred



aim_model = makeAIMNetModel()

def AIMNET_Predictor(np_img):
    predict = inference_img(aim_model, np_img)
    return predict