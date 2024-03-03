import os
import torch
import numpy as np

DATA_ROOT = "/scratch/jrottmay/data/rae_rans"
subset = "img_64x64"

import lightning.pytorch as pl
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split

class RAE(Dataset):
    """
    Custom RAE flow field dataset from a generic DOE process in a bounded design 
    space. Samples are post-processed to be tensors representing the region of 
    interest.
    """
    def __init__(self, 
        root_dir=f"{DATA_ROOT}",
        subset=f"{subset}",
        normalize=False
        ):

        self.root_dir = root_dir
        self.subset = subset
        self.file_list = os.listdir(os.path.join(root_dir, subset))
        self.normalize = normalize
        self.max = torch.Tensor([1.9963, 1.0358, 0.6987, 1.5153, 1.1502, 1.1357])
        self.min = torch.Tensor([0.0000, -0.6520, -0.5926,  0.0000, -1.7431, 0.0000])
        #self.max = torch.Tensor([1.3454, 3.7994, 1.9963, 1.0358, 0.6987, 1.5153, 1.1502, 1.1357])
        #self.min = torch.Tensor([0.0000, 0.0000, 0.0000, -0.6520, -0.5926,  0.0000, -1.7431, 0.0000])
        self.norm = torchvision.transforms.Normalize(mean=self.min, std=self.max-self.min)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # load_index is different to idx since some simulations are missing
        load_index = self.file_list[idx] 

        file = os.path.join(self.root_dir, self.subset, load_index, f"{load_index}.npz")
        sample = np.load(file)

        output = {}
        for key in sample.keys():
            if key in["var_names", "function_descriptors", "gradient_descriptors"]:
                output[key] = sample[key].tolist()
            elif key=="field":
                tmp = torch.permute(torch.as_tensor(sample[key], dtype=torch.float), (2,0,1))
                output[key] = self.norm(tmp) if self.normalize else tmp
            else:
                output[key] = torch.as_tensor(sample[key], dtype=torch.float)
        return output


class RAEDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, subset="img_128x128", split=0.9, num_workers: int = 0, normalize=False):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.num_workers=num_workers
        self.subset = subset
        self.normalize = normalize

    def prepare_data(self):
        None

    def setup(self, stage: str):
        full_data = RAE(subset=self.subset, normalize=self.normalize)
        l = len(full_data)
        lts = int(l * self.split)
        self.train, self.val = random_split(
            full_data, [lts, l-lts], generator=torch.Generator().manual_seed(42)
        )

        # Assign test dataset for use in dataloader(s)
        self.test = self.val

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        None     
