import os
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image


class Tod_dataset(Dataset):
    def __init__(self, path_dir, num_class=3, transform=None):
        self.transform = transform  # using transform in torch!
        self.path_dir = path_dir
        self.num_class = num_class
        self.color2cls = {
            0: 0, 127: 1, 255: 2
        }

        gt_name = sorted(os.listdir(os.path.join(path_dir, 'GTs')))
        self.gts = []
        for name in gt_name:
            if name.endswith('.png'):
                self.gts.append(name)
        input_name = sorted(os.listdir(os.path.join(path_dir, 'inputs')))
        self.inputs = []
        for name in input_name:
            if name.endswith('.png'):
                self.inputs.append(name)

        assert len(self.gts) == len(self.inputs)

    def __len__(self):
        return len(self.gts)
    
    def __getitem__(self, idx):
        input = Image.open(os.path.join(os.path.join(self.path_dir, 'inputs'), self.inputs[idx]))
        input = np.array(input)
        gt = Image.open(os.path.join(os.path.join(self.path_dir, 'GTs'), self.gts[idx]))
        gt = np.array(gt)
        gt_cls = np.vectorize(self.color2cls.get)(gt)
        eye = np.eye(self.num_class)
        gt_one_hot = eye[gt_cls]
        return {'input': input,
                'gt': gt_one_hot.transpose(2, 0, 1)}
