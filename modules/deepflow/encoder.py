import torch
import torch.nn as nn

from .res_layer import ResidualLayer
from ..distribution import DiagonalGaussianDistribution

class Encoder(nn.Module):

    def __init__(self, 
            input_shape=None,
            base_channels=16, 
            latent_dim=128, 
            image_layers=0,
            **kwargs):

        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_shape
        self.image_layers = image_layers
        self.base_channels = base_channels

        resblocks=[]
        resblocks.append(
            nn.Sequential(
                ResidualLayer(channels=self.input_dim[0], downsize=True),
                nn.Conv2d(self.input_dim[0], self.base_channels, kernel_size=(3,3), padding='same')
                )
            )
        for i in range(1,self.image_layers):
            resblocks.append(
                nn.Sequential(
                    ResidualLayer(channels=self.base_channels*(i), downsize=True),
                    nn.Conv2d(self.base_channels*(i), self.base_channels*(i+1), kernel_size=(3,3), padding='same')   
                )
            )
        self.mid = nn.Sequential(*resblocks)    
        self.conv = nn.Conv2d(self.base_channels * self.image_layers, self.latent_dim, kernel_size=(3,3), padding='same')  
        self.res_block_same = ResidualLayer(channels=self.latent_dim)
        
        wr, hr = (int(self.input_dim[1] / 2**self.image_layers), int(self.input_dim[2] / 2**self.image_layers))
        self.flatten = nn.Flatten(start_dim=1)# b c (h*w)
        
        self.z_mean = nn.Linear(wr*hr*self.latent_dim, wr*hr*self.latent_dim)
        self.z_var  = nn.Linear(wr*hr*self.latent_dim, wr*hr*self.latent_dim)
        self.reduced_dim = (self.latent_dim, wr, hr)

    def forward(self, inputs):
        b, c, w, h = inputs.shape
        z = inputs

        z = self.mid(z)

        z = self.conv(z)
        z = self.res_block_same(z)

        z = self.flatten(z)
        z_mean = self.z_mean(z)
        z_var = self.z_var(z)
        z_mean = torch.reshape(z_mean, (b, *self.reduced_dim))
        z_var = torch.reshape(z_var, (b, *self.reduced_dim))
        epsilon = torch.randn(z_mean.shape, device=z_mean.device)
        return z_mean, z_var, z_mean + torch.exp(0.5 * z_var) * epsilon