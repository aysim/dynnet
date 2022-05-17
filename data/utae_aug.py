"""
Adapted from: Semi-supervised Semantic Segmentation with Directional Context-aware Consistency (CAC) Implementation Adapted
Author: Xin Lai*, Zhuotao Tian*, Li Jiang, Shu Liu, Hengshuang Zhao, Liwei Wang, Jiaya Jia
License: MIT
"""
import random, math
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

class ToTensorScaled(object):
    '''Convert a Image to a CHW ordered Tensor, scale the range to [0, 1]'''
    def __call__(self, im):
        im = im.transpose((2, 0, 1))
        return torch.from_numpy(im)

mean = [1042.59240722656, 915.618408203125, 671.260559082031, 2605.20922851562]
std = [957.958435058593, 715.548767089843, 596.943908691406, 1059.90319824218]

scaletensor = ToTensorScaled()
normalize = transforms.Normalize(mean=mean, std=std)
def _scaleNormalize_image_stack(image_list):
    cur_images = []
    for i in range(image_list.shape[0]):
        cur_images.append(
            torch.unsqueeze(normalize(scaletensor(np.asarray(image_list[i, :, :, :], dtype=np.float32))), 0))

    image_stack = torch.cat(cur_images, dim=0)
    return image_stack

def _crop(image_list, label, crop_size, image_padding, ignore_index):
    # Padding to return the correct crop size
    if (isinstance(crop_size, list) or isinstance(crop_size, tuple)) and len(crop_size) == 2:
        crop_h, crop_w = crop_size
    elif isinstance(crop_size, int):
        crop_h, crop_w = crop_size, crop_size
    else:
        raise ValueError

    h, w = label.shape
    pad_h = max(crop_h - h, 0)
    pad_w = max(crop_w - w, 0)
    pad_kwargs = {
        "top": 0,
        "bottom": pad_h,
        "left": 0,
        "right": pad_w,
        "borderType": cv2.BORDER_CONSTANT, }
    if pad_h > 0 or pad_w > 0:
        label = cv2.copyMakeBorder(label, value=ignore_index, **pad_kwargs)
        for i in range(image_list.shape[0]):
            image_list[i, :, :, :] = cv2.copyMakeBorder(image_list[i, :, :, :], value=image_padding, **pad_kwargs)

    # Cropping
    h, w = label.shape
    cropped_img = []
    start_h = random.randint(0, h - crop_h)
    start_w = random.randint(0, w - crop_w)
    end_h = start_h + crop_h
    end_w = start_w + crop_w
    label = label[start_h:end_h, start_w:end_w]
    for i in range(image_list.shape[0]):
        cropped_img.append(np.expand_dims(image_list[i, :, :, :][start_h:end_h, start_w:end_w], axis=0))
    image_list = np.concatenate(cropped_img, axis=0)
    return image_list, label

def _flip(image_list, label):
    # Random H flip
    flipped_img = []
    if random.random() > 0.5:
        label = np.fliplr(label).copy()
        for i in range(image_list.shape[0]):
            image = np.fliplr(image_list[i, :, :, :]).copy()
            flipped_img.append(np.expand_dims(image, axis=0))
        image_list = np.concatenate(flipped_img, axis=0)
    return image_list, label

def _resize(image_list, label, base_size, scale, bigger_side_to_base_size=True):
    if isinstance(base_size, int):
        h, w = label.shape
        resized_img = []

        if scale:
            longside = random.randint(int(base_size * 0.5), int(base_size * 2.0))
        else:
            longside = base_size

        if bigger_side_to_base_size:
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (
            int(1.0 * longside * h / w + 0.5), longside)
        else:
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h < w else (
            int(1.0 * longside * h / w + 0.5), longside)

        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        for i in range(image_list.shape[0]):
            image = cv2.resize(image_list[i, :, :, :], (w,h), interpolation=cv2.INTER_CUBIC)
            resized_img.append(np.expand_dims(image, axis=0))
        image_list = np.concatenate(resized_img, axis=0)
        return image_list, label

    else:
        raise ValueError

def val_augmentation(image_list, label, scale, base_size=None):
    if base_size is not None:
        image_list, label = _resize(image_list, label, base_size, scale)
        image_list = _scaleNormalize_image_stack(image_list)
        return image_list, label

    image_list = _scaleNormalize_image_stack(image_list)
    return image_list, label

def train_augmentation(image_list, label, scale, base_size, crop_size, image_padding, ignore_index, flip):
    if base_size is not None:
        image_list, label = _resize(image_list, label, base_size, scale)

    if crop_size is not None:
        image_list, label = _crop(image_list, label, crop_size=crop_size, image_padding=image_padding, ignore_index=ignore_index)
    if flip:
        image_list, label = _flip(image_list, label)
    return _scaleNormalize_image_stack(image_list), label