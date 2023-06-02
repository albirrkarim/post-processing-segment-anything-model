"""
Deep Automatic Natural Image Matting [IJCAI-21]
Main test file.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/AIM
Paper link : https://www.ijcai.org/proceedings/2021/111

"""
# import os
import torch
# import cv2
# import argparse
import numpy as np
# from tqdm import tqdm
# from PIL import Image
from skimage.transform import resize
from torchvision import transforms
# import logging
# import json

import sys

sys.path.append('../algorithms/AIM/core')


from config import *
from util import *
from evaluate import *
from network.AimNet import AimNet


def makeAIMNetModel():
    # print('*********************************')
    # print(f'Loading model: {model_path}')
    # print(f'Test stategy: {args.test_choice}')
    # print(f'Test dataset: {args.dataset_choice}')
    model_path = '/Users/susanto/Documents/Proyek/best-remove-background/models/aimnet_pretrained_matting.pth'
    
    model = AimNet()

    if torch.cuda.device_count() == 0:
        # print(f'Running on CPU...')
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        # print(f'Running on GPU with CUDA as {cuda}...')
        ckpt = torch.load(model_path)

    model.load_state_dict(ckpt['state_dict'], strict=True)
    
    model = model.cpu()

    model.eval()
    
    return model

def evaluate_single_img(predict, alpha):
    sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(
        predict, alpha)

    conn_diff = compute_connectivity_loss_whole_image(predict, alpha)

    grad_diff = compute_gradient_whole_image(predict, alpha)

    return {
        "sad_diff": sad_diff,
        "mse_diff": mse_diff,
        "mad_diff": mad_diff,
        "conn_diff": conn_diff,
        "grad_diff": grad_diff
    }


def inference_once(model, scale_img, scale_trimap=None):
    pred_list = []
    cuda = False

    if cuda:
        tensor_img = torch.from_numpy(scale_img.astype(
            np.float32)[:, :, :]).permute(2, 0, 1).cuda()
    else:
        tensor_img = torch.from_numpy(scale_img.astype(np.float32)[
                                      :, :, :]).permute(2, 0, 1)

    input_t = tensor_img
    input_t = input_t/255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    input_t = normalize(input_t)
    input_t = input_t.unsqueeze(0)
    # forward
    pred_global, pred_local, pred_fusion = model(input_t)
    pred_global = pred_global.data.cpu().numpy()
    pred_global = gen_trimap_from_segmap_e2e(pred_global)
    pred_local = pred_local.data.cpu().numpy()[0, 0, :, :]
    pred_fusion = pred_fusion.data.cpu().numpy()[0, 0, :, :]

    return pred_global, pred_local, pred_fusion


def inference_img(model, img):
    test_choice = "HYBRID"

    h, w, c = img.shape
    new_h = min(MAX_SIZE_H, h - (h % 32))
    new_w = min(MAX_SIZE_W, w - (w % 32))

    if test_choice == 'HYBRID':
        global_ratio = 1/4
        local_ratio = 1/2

        resize_h = int(h*global_ratio)
        resize_w = int(w*global_ratio)
        new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
        new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))

        scale_img = resize(img, (new_h, new_w))*255.0
        pred_coutour_1, pred_retouching_1, pred_fusion_1 = inference_once(
            model, scale_img)
        pred_coutour_1 = resize(pred_coutour_1, (h, w))*255.0

        resize_h = int(h*local_ratio)
        resize_w = int(w*local_ratio)
        new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
        new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
        scale_img = resize(img, (new_h, new_w))*255.0
        pred_coutour_2, pred_retouching_2, pred_fusion_2 = inference_once(
            model, scale_img)
        pred_retouching_2 = resize(pred_retouching_2, (h, w))

        pred_fusion = get_masked_local_from_global_test(
            pred_coutour_1, pred_retouching_2)

        return pred_fusion
    else:
        resize_h = int(h/4)
        resize_w = int(w/4)
        new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
        new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
        scale_img = resize(img, (new_h, new_w))*255.0
        pred_global, pred_local, pred_fusion = inference_once(model, scale_img)
        pred_local = resize(pred_local, (h, w))
        pred_global = resize(pred_global, (h, w))*255.0
        pred_fusion = resize(pred_fusion, (h, w))

        return pred_fusion