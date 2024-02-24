import torch
import torch.nn as nn

from ..misc.downsample import Downsample


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class ResidualLayer(nn.Module):
    def __init__(self,
            channels=None,
            downsize=False, 
            **kwargs):

        super().__init__(**kwargs)
        self.channels = channels
        self.downsize = downsize
        self.c2_1 = nn.Conv2d(self.channels, self.channels, kernel_size=(3,3), padding='same')
        self.c2_2 = nn.Conv2d(self.channels, self.channels, kernel_size=(3,3), padding='same')
        self.activation = nn.LeakyReLU()
        self.post = Downsample(self.channels) if downsize else nn.Identity()
    
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