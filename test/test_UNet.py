#!/home/jrottmay/.conda/envs/su/bin/python3.10
import unittest
import numpy as np
import torch
import torch.nn as nn
import sys
import torchvision

torch.manual_seed(0)
iterations=2

class Test_UNet_MNIST(unittest.TestCase):

    def test_unet_on_mnist(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device=='cpu' and False:
            print("Device is 'cpu'. For testing make sure to have gpu available")
            sys.exit()

        train_data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('~/data', 
                transform=torchvision.transforms.ToTensor(), 
                download=True),
            batch_size=128,
            shuffle=True)

        from modules import UNet

        model = UNet(1, 32,
            channel_mults=(1, 2, 4),
            num_classes=10,
            num_groups=32,
            )

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        loss_function = nn.MSELoss()

        x, y = next(iter(train_data))
        x = x.to(device)
        y = y.to(device)

        x_p = model(x, y=y)
        initial_loss = loss_function(x_p, x)
        for i in range(1,iterations+1):
            model.train()
            x_p = model(x, y=y)
            loss = loss_function(x_p, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        x_p = model(x, y=y)
        loss = loss_function(x_p, x)
        assert(not np.isnan(loss.detach().numpy()))
        assert(loss < initial_loss)
