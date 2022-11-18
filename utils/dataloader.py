#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : dataloader
# @Date : 2022-11-18-12-51
# @Project : aidant_retinal-disease
# @Author : seungmin

import os

from PIL import Image
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import pydicom

from torch.utils.data import DataLoader
import torchvision.transforms as T

np.random.seed(0)


class MyDataset(object):

    def __init__(self, df, transform=None):
        self.file_list = df
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        print(img_path)
        if img_path.split('.')[-1] == 'dcm':  # dcm인 경우
            img = pydicom.dcmread(img_path)
            img = img.pixel_array.astype(float)
            img = (np.maximum(img, 0) / img.max()) * 255.0
            img = np.uint8(img)
        else:  # png/jpeg 인 경우
            # img = Image.open(img_path).resize((512, 512)).convert('RGB')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (512, 512))

        # img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        # img_clahe = img_yuv.copy()
        # clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        # img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])
        # img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2RGB)

        img_avg = np.mean(img)
        img = cv2.resize(img, (512, 512))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 512 / 30), -4, 128)

        gamma = 1.0
        if img_avg < 10.0:
            gamma = 0.5
        elif 10.0 <= img_avg < 20:
            gamma = 0.6
        elif 20 <= img_avg < 30:
            gamma = 0.7
        elif 30 <= img_avg < 40:
            gamma = 0.8
        elif 40 <= img_avg < 46.5:
            gamma = 0.9

        img = Image.fromarray(img)
        img = T.functional.adjust_gamma(img, gamma=gamma)

        img = np.array(img)

        if self.transform is not None:
            img = self.transform(img)

        if bool(img_path.find('amd') + 1 or img_path.find('AMD') + 1):
            label = 1  # amd인 경우 1을 반환
        elif bool(img_path.find('dr') + 1):
            label = 2  # dr인 경우 2을 반환
        elif bool(img_path.find('glaucoma') + 1):
            label = 3  # glaucoma인 경우 3을 반환
        else:
            label = 0  # normal

        return img, label


def make_datapath_list(root_path):
    data_list = [root_path]
    return data_list  # , txt_list


class MyInferDatasetWrapper(object):

    def __init__(self, batch_size, num_workers, test_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_path = test_path

    def get_test_loaders(self):
        no_augment = self._get_transform()

        test_df = make_datapath_list(self.test_path)
        test_dataset = MyDataset(test_df, transform=no_augment)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 num_workers=self.num_workers, drop_last=True, shuffle=False,
                                 pin_memory=True)

        return test_loader

    def _get_transform(self):
        return T.Compose([T.ToPILImage(),
                          T.Resize(512),
                          T.ToTensor(),
                          # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])