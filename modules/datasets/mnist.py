import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader

import torch
from torchvision.datasets import MNIST
from torchvision import transforms

from torch.utils.data import Dataset
from torchvision import datasets

class ModifiedMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.mnist_dataset = datasets.MNIST(root, train=train, transform=transform, download=download)

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        image, label = self.mnist_dataset[index]
        label = torch.nn.functional.one_hot(torch.as_tensor(label), num_classes=10).float()
        return {"image": image, "label": label}  

class MNISTDataModule(pl.LightningDataModule):
    
    def __init__(self, data_dir: str = "/scratch/jrottmay/data", batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers=num_workers
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        # download
        ModifiedMNIST(self.data_dir, train=True, download=True, transform=self.transform)
        ModifiedMNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == 'validate':
            mnist_full = ModifiedMNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = ModifiedMNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = ModifiedMNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        None