import os
import random
import h5py
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image


class Tod_dataset(Dataset):
    def __init__(self, path_dir, transform=None):
        self.transform = transform  # using transform in torch!
        self.path_dir = path_dir

        gt_name = sorted(os.listdir(os.path.join(path_dir, 'gt')))
        self.gts = []
        for name in gt_name:
            if name.endswith('.tif'):
                self.gts.append(name)
        input_name = sorted(os.listdir(os.path.join(path_dir, 'input')))
        self.inputs = []
        for name in input_name:
            if name.endswith('.tif'):
                self.inputs.append(name)

        assert len(self.gts) == len(self.inputs)

    def __len__(self):
        return len(self.gts)
    
    def __getitem__(self, idx):
        input = Image.open(os.path.join(os.path.join(self.path_dir, 'input'), self.inputs[idx]))
        input = np.array(input)
        gt = Image.open(os.path.join(os.path.join(self.path_dir, 'gt'), self.gts[idx]))
        gt = np.array(gt)
        return {'input': input,
                'gt': gt}
