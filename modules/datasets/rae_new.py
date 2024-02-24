import os
import torch
import numpy as np

DATA_ROOT = "/scratch/jrottmay/data/rae_rans"
subset = "img_64x64"

import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, random_split

class RAE(Dataset):
    """
    Custom RAE flow field dataset from a generic DOE process in a bounded design 
    space. Samples are post-processed to be tensors representing the region of 
    interest.
    """
    def __init__(self, 
        root_dir=f"{DATA_ROOT}/{subset}",
        normalize=False # Not yet supported
        ):

        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # load_index is different to idx since some simulations are missing
        load_index = self.file_list[idx] 

        file = os.path.join(self.root_dir, load_index, f"{load_index}.npz")
        sample = np.load(file)

        output = {}
        for key in sample.keys():
            if key=="var_names":
                output[key] = sample[key].tolist()
            elif key=="field":
                output[key] = torch.permute(torch.as_tensor(sample[key], dtype=torch.float), (2,0,1))
            else:
                output[key] = torch.as_tensor(sample[key], dtype=torch.float)
        return output


class HeinDoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, split=0.9, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.num_workers=num_workers

    def prepare_data(self):
        None

    def setup(self, stage: str):
        full_data = HeinDo()
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
