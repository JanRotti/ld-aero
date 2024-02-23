
import os
import torch
import pandas as pd
import numpy as np

import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage import zoom
from skimage.measure import block_reduce

# PATH
DATA_ROOT = "/scratch/jrottmay/data"

class RAE(Dataset):
    """
    Custom RAE flow field dataset from a generic DOE process in a bounded design 
    space. Samples are post-processed to be tensors representing the region of 
    interest.
    """
    def __init__(self, 
        root_dir=f"{DATA_ROOT}/rae_rans/results", 
        flow_dir="flow", 
        label_file="misc/dvs.csv",
        data_labels="misc/var_names.txt",
        transform=None, 
        normalize=False
        ):

        self.root_dir = root_dir
        self.flow_dir = flow_dir
        self.labels = np.genfromtxt(os.path.join(root_dir, label_file), delimiter=",")
        self.channel_names = pd.read_csv(os.path.join(root_dir, data_labels), header=None).values.flatten()
        
        self.transform = transform
        self.normalize = normalize
    
        self.file_list = []
        
        self.max = torch.tensor([1.3474, 3.8084, 2.0157, 1.0562, 0.8013, 1.5173, 1.1547, 1.1359])
        self.min = torch.tensor([0.0000,  0.0000,  0.0000, -0.6628, -0.7008,  0.0000, -1.7557,  0.0000])
        
        for file in os.listdir(os.path.join(root_dir, flow_dir)):
            if not file.endswith(".npy"):
                continue
            self.file_list.append(os.path.join(root_dir, flow_dir, file))
        assert len(self.file_list) > 0, "Number samples must be greater than 0."
        assert len(self.file_list) == len(self.labels), "Number of samples and labels must match."
        
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
        
        sample = sample.permute(2, 0, 1)
        # Return sample and label as x, y 
        return {"image": sample, "label": label}


class RAEDataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size: int = 32, split=0.9, test_split=0.05, normalize=True, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.normalize = normalize
        self.num_workers=num_workers
        self.transform = example_transform

    def prepare_data(self):
        None

    def setup(self, stage: str):
        rae_full = RAE(transform=self.transform, normalize=self.normalize)
        l = len(rae_full)
        lts = int(l * self.split)
        self.rae_train, self.rae_val = random_split(
            rae_full, [lts, l-lts], generator=torch.Generator().manual_seed(42)
        )

        # Assign test dataset for use in dataloader(s)
        self.rae_test = self.rae_val

    def train_dataloader(self):
        return DataLoader(self.rae_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.rae_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.rae_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.rae_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        None
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

