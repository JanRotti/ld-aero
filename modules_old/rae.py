import os
import torch
import pandas as pd

import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage import zoom
from skimage.measure import block_reduce

class RAEFlowFieldsDataset(Dataset):
    """
    RAE flow field dataset from a generic DOE process in a bounded design space.
    Samples are post-processed to be tensors representing the region of interest.

    * Configuration and details can be found in '/home/jrottmay/data/rae_rans' and
    '/home/jrottmay/data_generation/rae_rans_doe'.
    """
    def __init__(self, 
        root_dir="/home/jrottmay/data/rae_rans/results", 
        flow_dir="flow", 
        label_file="misc/dvs.csv", 
        transform=None, 
        normalize=False
        ):

        self.root_dir = root_dir
        self.flow_dir = flow_dir
        self.transform = transform
        self.normalize = normalize

        self.labels = np.genfromtxt(os.path.join(root_dir, label_file), delimiter=",")
        self.file_list = []
        
        self.max = torch.tensor([1.3474, 3.8084, 2.0157, 1.0562, 0.8013, 1.5173, 1.1547, 1.1359])
        self.min = torch.tensor([0.0000,  0.0000,  0.0000, -0.6628, -0.7008,  0.0000, -1.7557,  0.0000])
        
        for file in os.listdir(os.path.join(root_dir, flow_dir)):
            if not file.endswith(".npy"):
                continue
            self.file_list.append(os.path.join(root_dir, flow_dir, file))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = os.path.join(self.root_dir, self.flow_dir, self.file_list[idx])
        label = self.labels[idx]
        sample = np.load(file)
        
        # Convert to torch tensor
        sample = torch.from_numpy(sample)
        # Convert from double to single prec
        sample = sample.type(torch.float)

        # Apply transformation if available
        if self.transform:
            sample = self.transform(sample)

        # Normalize if requested
        if self.normalize:
            sample = sample.sub(self.min[None, None, :]).div(self.max[None, None, :] - self.min[None, None, :])
        
        # Return sample and label as x, y 
        return sample, label


"""
    downsamples the RAE dataset samples to 64x64x8
"""
def example_transform(sample):
    # Make sample square
    w, h, c = sample.shape
    cut = min(w, h) 
    # Use all channels
    sample = sample[:cut,:cut,:] 
    # Desired dim
    ddim = 64
    zoom_factor = ddim / cut
    # Downsample 
    channels = []
    for i in range(c):
        channel = zoom(sample[:, :, i], zoom_factor)
        channels.append(channel)

    sample = np.stack(channels, axis=-1)
    sample = torch.from_numpy(sample)
    return sample