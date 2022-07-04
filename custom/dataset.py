import matplotlib.pyplot as plt
import time
import numpy as np
import argparse
import math
import os
from glob import glob
import random
from PIL import Image
from tqdm.auto import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms
from PIL import Image

from .utils import *
from torchvision import utils
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pydicom


class RealDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 split='train',
                 transform=None,
                 ):
        self.split          = split
        self.transform      = transform
        self.imgs = glob(os.path.join(data_path, "*", "*.png"))
        n_imgs = len(self.imgs)
        
        n_train = int(n_imgs * 0.8)
        n_valid = int(n_imgs * 0.1)
        n_test  = n_imgs - (n_train + n_valid)

        self.imgs = {"train": self.imgs[:n_train],
                     "valid": self.imgs[n_train: n_train + n_valid],
                     "test" : self.imgs[-n_test:]}

        self.imgs = self.imgs[split]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        
        if self.transform:
            img = self.transform(img)

        return img

class BrainCT(torch.utils.data.Dataset):
    def __init__(self, query_save_dir, transform, reverse=True):
        self.transform  = transform

        self.imgs      = sorted(glob(os.path.join(query_save_dir, 'png', '*.png')), reverse = reverse)[:32]
        self.fNames    = [img.split('/')[-1] for img in self.imgs]
        self.imgs      = [Image.open(img) for img in self.imgs]

        bet_npy        = os.path.join(query_save_dir, 'BrainExtraction', 'bet.npy')
        if os.path.exists(bet_npy): bet_np = np.load(bet_npy)
        else:                       bet_np = np.ones([len(self.imgs),512,512])
        if reverse:
            self.bet_masks = torch.from_numpy(np.flip(bet_np, 0).copy())[:32]
        else:
            self.bet_masks = torch.from_numpy(bet_np)[:32]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        fName, img, bet_mask = self.fNames[idx], self.imgs[idx], self.bet_masks[idx]
        if self.transform:
            img = self.transform(img)
        return idx, fName, img, bet_mask
