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

        self.resblocks=[]
        for i in range(self.image_layers):
            self.resblocks.append(
                ResidualLayer(channels=self.base_channels*(i+1), downsize=True)
            )
        self.conv = nn.Conv2D(self.base_channels * self.image_layers, self.latent_dim, kernel_size=(3,3), padding='same')  
        self.res_block_same = ResidualLayer(channels=self.latent_dim)
        
        wr, hr = (int(self.input_dim[1] / 2**self.image_layers), int(self.input_dim[2] / 2**self.image_layers))
        self.flatten = nn.Flatten(start_dim=-2)# b c (h*w)
        
        self.z_mean = nn.Linear(wr*hr, self.latent_dim)
        self.z_var  = nn.Linear(wr*hr, self.latent_dim)
    

    def forward(self, inputs):
        b, c, w, h = inputs.shape
        z = inputs

        for i in range(self.image_layers):
            z = self.resblocks[i](z)

        z = self.conv(z)
        z = self.res_block_same(z)

        z = self.flatten(z)
        tmp_mean = torch.zeros((b, self.latent_dim)) # reuse weights!
        tmp_var = torch.zeros((b, self.latent_dim)) # reuse weights!
        
        for i in range(self.latent_dim):
            tmp_mean += self.z_mean(z[:,i,:])
            tmp_var += self.z_var(z[:,i,:])
        
        output = DiagonalGaussianDistribution(torch.cat((tmp_mean, tmp_var), dim=1))
        return output.sample()