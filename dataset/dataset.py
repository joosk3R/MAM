import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import nibabel as nib
import numpy as np
import kornia as K

import os
import json
from math import cos, sin, atan, radians, degrees
from natsort import natsorted
from einops import rearrange


import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import kornia.augmentation as K

class SNHUMipDataset(Dataset):
    def __init__(self, images, dims, org,affine, augmentation=True, degree=(0, 0), translate=(0.2, 0.2), p=0.5):
        """
        images: 리스트 또는 배열, 각각의 2D MIP 이미지 데이터
        dims: 리스트, 각 이미지의 dim 정보
        resizes: 리스트, 각 이미지의 resize 정보
        augmentation: 데이터 증강 여부
        degree: 회전 각도 범위
        translate: 평행 이동 범위
        p: 증강 확률
        """
        self.images = images
        self.dims = dims
        self.augmentation = augmentation
        self.degree = degree
        self.translate = translate
        self.p = p
        self.org_dcm = org
        self.org_dcm = self.org_dcm[None]
        self.affine = affine
        self.initial_patient_position, self.voxel_spacing = affine
        self.initial_patient_position = np.array(self.initial_patient_position)[None]
        self.voxel_spacing = np.array(self.voxel_spacing)[None]
    
    def Affine_transformation(self, img, degree=(0, 0), translate=(0, 0), p=1, padding_mode='zeros'):
        '''
        img: (c(1), d, h)
        '''
        Affine = K.augmentation.AugmentationSequential(
            K.augmentation.RandomAffine(degrees=degree, translate=translate, p=p, padding_mode=padding_mode),
            data_keys=['input'],
        )
        T_img = Affine(img)
        return T_img[0]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        dim = self.dims[idx]
        org_dcm = self.org_dcm[idx]
        initial_patient_position = self.initial_patient_position[idx]
        voxel_spacing = self.voxel_spacing[idx]

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) # C(1) H D

        if self.augmentation:
            img_tensor = self.Affine_transformation(img_tensor, degree=self.degree, translate=self.translate, p=self.p, padding_mode='border')

        return img_tensor, dim ,org_dcm, initial_patient_position, voxel_spacing # (c(1), d, h), dim, resize