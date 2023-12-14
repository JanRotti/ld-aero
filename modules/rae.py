import os
import torch
import pandas as pd

import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class RAEFlowFieldsDataset(Dataset):
    """
    RAE flow field dataset from a generic DOE process in a bounded design space.
    Samples are post-processed to be tensors representing the region of interest.

    * Configuration and details can be found in '/home/jrottmay/data/rae_rans' and
    '/home/jrottmay/data_generation/rae_rans_doe'.
    """
    def __init__(self, root_dir="/home/jrottmay/data/rae_rans_doe_large_2", flow_dir="post", label_file="dvs.csv", transform=None, normalize=False):
        self.root_dir = root_dir
        self.flow_dir = flow_dir
        self.transform = transform
        self.normalize = normalize
        self.labels = np.genfromtxt(os.path.join(root_dir, label_file), delimiter=",")
        self.file_list = []
        self.max = torch.tensor([1.3243, 3.7887, 1.9563, 1.0714, 0.6163, 1.4819, 1.0763, 1.2320])
        self.min = torch.tensor([-0.1237, -0.3570, -0.1868, -0.5965, -0.4538, -0.1331, -1.7783, -0.1191])
        for file in os.listdir(os.path.join(root_dir, flow_dir)):
            if not file.endswith(".npy"):
                continue
            self.file_list.append(os.path.join(root_dir, flow_dir, file))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = os.path.join(self.root_dir,self.flow_dir, self.file_list[idx])
        label = self.labels[idx]
        sample = np.load(file)

        if self.transform:
            sample = self.transform(sample)
        
        if self.normalize:
            sample = sample.sub(self.min[:,None,None]).div(self.max[:,None,None] - self.min[:,None,None])
        sample = sample 
        return sample, label