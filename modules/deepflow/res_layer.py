import torch
import torch.nn as nn

from ..misc.downsample import Downsample

class ResidualLayer(nn.Module):
   def __init__(self,
            channels=None,
            downsize=False, 
            **kwargs):

        super().__init__(**kwargs)
        self.channels = channels
        self.downsize = downsize
        self.c2_1 = nn.Conv2D(self.channels, self.channels, kernel_size=(3,3), padding='same')
        self.c2_2 = nn.Conv2D(self.channels, self.channels, kernel_size=(3,3), padding='same')
        self.activation = nn.LeakyReLU()
        self.post = Downsample(self.channels) if downsize else None

    def forward(self, inputs):
        x_skip = inputs
        x = inputs
        x = self.c2_1(x)
        x = self.activation(x)
        x = self.c2_2(x)
        x += x_skip
        if self.downsize:
            x = self.post(x)
        return x