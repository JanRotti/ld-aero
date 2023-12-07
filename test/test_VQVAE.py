#!/home/jrottmay/.conda/envs/su/bin/python3.10
import unittest
import numpy as np
import torch
import torch.nn as nn
import sys
import torchvision

torch.manual_seed(0)
iterations=2

class Test_VQVAE_MNIST(unittest.TestCase):

    def test_vqvae_on_mnist(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device=='cpu' and False:
            print("Device is 'cpu'. For testing make sure to have gpu available")
            sys.exit()

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),  # Resize to 128x128
            torchvision.transforms.ToTensor(),  # Convert PIL image to tensor
        ])

        train_data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('~/data', 
                transform=transform, 
                download=True),
            batch_size=128,
            shuffle=True)

        from modules import VQVAE

        model = VQVAE(1, 32, 10,
            channel_mults=(1, 2, 4),
            attention_resolutions=(0, 1),
            num_res_blocks=1,
            norm="gn",
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
