#!/home/jrottmay/.conda/envs/su/bin/python3.10
import unittest
from unittest.mock import patch, Mock

import sys
import os
import numpy as np
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torchvision
torch.manual_seed(0)

class Test_train(unittest.TestCase):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cpu' and False:
        print("Device is 'cpu'. For testing make sure to have gpu available")
        sys.exit()

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 10),
        nn.Sigmoid(),
    )

    optim = torch.optim.Adam(model.parameters(), lr=1e-5)

    train_data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('~/data', 
                transform=torchvision.transforms.ToTensor(), 
                download=True),
            batch_size=128,
            shuffle=True)
    
    @patch('sys.stderr', new=unittest.mock.Mock())
    def test_train_minimal(self):
        from modules import train, to_categorical
        loss_function = lambda m, x, y: {'loss':F.cross_entropy(m(x), torch.from_numpy(to_categorical(y, 10)).float())}
        train(self.model, self.optim, self.train_data, loss_function,iterations=10,checkpoint_rate=10, log_rate=10)

    @patch('sys.stderr', new=unittest.mock.Mock())
    def test_train_extensive(self):
        from modules import train, to_categorical
        loss_function = lambda m, x, y: {'loss':F.cross_entropy(m(x), torch.from_numpy(to_categorical(y, 10)).float())}
        train(self.model, self.optim, self.train_data, loss_function,
            iterations=10, checkpoint_rate=10, log_rate=10, log_to_wandb=True, 
            model_checkpoint="log/unnamed-testing-iteration-10-model.pth",
            optim_checkpoint="log/unnamed-testing-iteration-10-optim.pth")