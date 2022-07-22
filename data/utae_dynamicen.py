import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datetime import datetime
from utils import palette
from data.utae_aug import train_augmentation, val_augmentation
import random
from PIL import Image

def undo_normalize_scale(im):
    mean = [1042.59240722656, 915.618408203125, 671.260559082031, 2605.20922851562]
    std = [957.958435058593, 715.548767089843, 596.943908691406, 1059.90319824218]
    im = im * std + mean
    array_min, array_max = im.min(), im.max()
    im = (im - array_min) / (array_max - array_min)
    im *= 255.0
    return im.astype(np.uint8)

def tens2image(im):
    tmp = np.squeeze(im.numpy())
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))

class ToTensorScaled(object):
    '''Convert a Image to a CHW ordered Tensor, scale the range to [0, 1]'''
    def __call__(self, im):
        im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
        return torch.from_numpy(im)

class DynamicEarthNet(Dataset):
    def __init__(self, root, mode, type, reference_date="2018-01-01", crop_size=512, num_classes=6, ignore_index=-1):
        """
        Args:
            root: the root of the folder which contains planet imagery and labels
            mode: train/val/test -- selects the splits
            type: single/weekly/daily -- selects the time-period you want to use
            reference_date: for positional encoding defaults:2018-01-01
            crop_size: crop size default:1024x1024
            num_classes: for DynamicEarthNet numclasses: 6
            ignore_index: default:-1
        """
        self.mode = mode
        self.type = type
        self.resize = crop_size
        self.root = root
        self.ignore_index = ignore_index
        self.files = []

        self.num_classes = num_classes
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.scalete = ToTensorScaled()
        self.mean = [1042.59240722656, 915.618408203125, 671.260559082031, 2605.20922851562]
        self.std = [957.958435058593, 715.548767089843, 596.943908691406, 1059.90319824218]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.set_files()

    def set_files(self):
        self.file_list = os.path.join(self.root, "dynnet_training_splits", f"{self.mode}" + ".txt")
        print (self.file_list)
        file_list = [line.rstrip().split(' ') for line in tuple(open(self.file_list, "r"))]
        self.files, self.labels, self.year_months = list(zip(*file_list))

        if self.type == 'daily':
            self.all_days = list(range(len(self.files)))

            for i in range(len(self.files)):
                self.planet, self.day = [], []
                date_count = 0
                for _, _, infiles in os.walk(os.path.join(self.root, self.files[i][1:])):
                    for infile in sorted(infiles):
                        if infile.startswith(self.year_months[i]):
                            self.planet.append(os.path.join(self.files[i], infile))
                            self.day.append((datetime(int(str(infile.split('.')[0])[:4]), int(str(infile.split('.')[0][5:7])),
                                                  int(str(infile.split('.')[0])[8:])) - self.reference_date).days)
                            date_count += 1
                self.all_days[i] = list(zip(self.planet, self.day))
                self.all_days[i].insert(0, date_count)

        else:
            self.planet, self.day = [], []
            if self.type == 'weekly':
                self.dates = ['01', '05', '10', '15', '20', '25']
            elif self.type == 'single':
                self.dates = ['01']

            for i, year_month in enumerate(self.year_months):
                for date in self.dates:
                    curr_date = year_month + '-' + date
                    self.planet.append(os.path.join(self.files[i], curr_date + '.tif'))
                    self.day.append((datetime(int(str(curr_date)[:4]), int(str(curr_date[5:7])),
                                                  int(str(curr_date)[8:])) - self.reference_date).days)
            self.planet_day = list(zip(*[iter(self.planet)] * len(self.dates), *[iter(self.day)] * len(self.dates)))


    def load_data(self, index):
        cur_images, cur_dates = [], []
        if self.type == 'daily':
            for i in range(1, self.all_days[index][0]+1):
                img = rasterio.open(os.path.join(self.root, self.all_days[index][i][0][1:]))
                red = img.read(3)
                green = img.read(2)
                blue = img.read(1)
                nir = img.read(4)
                image = np.dstack((red, green, blue, nir))
                cur_images.append(np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)) # np.array already\
                cur_dates.append(self.all_days[index][i][1])

            image_stack = np.concatenate(cur_images, axis=0)
            dates = torch.from_numpy(np.array(cur_dates, dtype=np.int32))
            label = rasterio.open(os.path.join(self.root, self.labels[index][1:]))
            label = label.read()
            mask = np.zeros((label.shape[1], label.shape[2]), dtype=np.int32)

            for i in range(self.num_classes + 1):
                if i == 6:
                    mask[label[i, :, :] == 255] = -1
                else:
                    mask[label[i, :, :] == 255] = i

            return (image_stack, dates), mask

        else:
            for i in range(len(self.dates)):
                # read .tif
                img = rasterio.open(os.path.join(self.root, self.planet_day[index][i][1:]))
                red = img.read(3)
                green = img.read(2)
                blue = img.read(1)
                nir = img.read(4)
                image = np.dstack((red, green, blue, nir))
                cur_images.append(np.expand_dims(np.asarray(image, dtype=np.float32), axis=0))   # np.array already\
            image_stack = np.concatenate(cur_images, axis=0)
            dates = torch.from_numpy(np.array(self.planet_day[index][len(self.dates):], dtype=np.int32))
            label = rasterio.open(os.path.join(self.root, self.labels[index][1:]))
            label = label.read()
            mask = np.zeros((label.shape[1], label.shape[2]), dtype=np.int32)

            for i in range(self.num_classes+1):
                if i == 6:
                    mask[label[i, :, :] == 255] = -1
                else:
                    mask[label[i, :, :] == 255] = i

            return (image_stack, dates), mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        padding = (np.array(self.mean)).tolist()
        (images, dates), label = self.load_data(index)
        base_size = label.shape[1]
        if 'val' in self.mode or 'test' in self.mode:
            images, label = val_augmentation(images, label, scale=False, base_size=None)
        else:
            images, label = train_augmentation(images, label, scale=True, base_size=base_size, crop_size=self.resize, flip=True)
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return (images, dates), label


