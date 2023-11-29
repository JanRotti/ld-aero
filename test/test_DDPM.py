#!/home/jrottmay/.conda/envs/su/bin/python3.10
import unittest
import numpy as np
import torch
import torch.nn as nn
import sys
import torchvision

torch.manual_seed(0)
iterations=2

class Test_DDPM_MNIST(unittest.TestCase):

    def test_ddpm_on_mnist(self):
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

        from modules import UNet, GaussianDiffusion, generate_linear_schedule

        schedule = generate_linear_schedule(10, 1e-4, 2e-2)

        unet = UNet(1, 32,
            channel_mults=(1, 2, 4),
            num_classes=10,
            num_groups=32,
            )

        model = GaussianDiffusion(unet, (28, 28), 1, 10, schedule)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        x, y = next(iter(train_data))
        x = x.to(device)
        y = y.to(device)

        initial_loss = model(x, y=y)
        for i in range(1,iterations+1):
            model.train()
            loss = model(x, y=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = model(x, y=y)
        assert(not np.isnan(loss.detach().numpy()))
        assert(loss < initial_loss)

if __name__ == '__main__':
    unittest.main()