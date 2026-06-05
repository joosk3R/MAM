import os, json
from operator import itemgetter
from pydicom.filereader import read_file
from dataset.dataset import SNHUMipDataset
from scipy.ndimage import affine_transform
from scipy.ndimage import zoom
from sklearn.mixture import GaussianMixture
import scipy.stats
from torch.utils.data import Dataset, DataLoader
import torch
import nibabel as nib
from torch.utils.data import DataLoader
from einops import rearrange 
from VGG import VGG16_depth4_fc1_concat
from natsort import natsorted
from operator import itemgetter
import copy

from utils import generate_gaussian_heatmap, resize_img

from monai.transforms import (
    Compose,
    ToTensord,
    NormalizeIntensityd,
    AddChanneld,
)
import os
import numpy as np
import torch
from model.detection_and_metal_classification_model import HourGlass3D
from utils import hadamard_product
from unet import UNet3D
from config import *

# 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")