import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels: int, channel_multipliers: list=[1], base_channels: int=16, bias: bool=False, **kwargs):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.bias = bias
        self.final_activation = nn.Sigmoid() # Probability
        self.channel_multipliers = channel_multipliers

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels * channel_multipliers[0], 4, 2, 1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True)
        )
        layers = []
        for c, cp in zip(channel_multipliers, channel_multipliers[1:]):
            layers.append(nn.Conv2d(base_channels * c, base_channels * cp, 4, 2, 1, bias=bias))
            layers.append(nn.BatchNorm2d(base_channels * cp))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.mid = nn.Sequential(*layers)
        
        self.flatten = nn.Flatten()
        self.final_linear = None

    def forward(self, input):
        b, c, h, w = input.shape
        if self.final_linear is None:
            step_down = 2**len(self.channel_multipliers)
            dim = (h // step_down) * (w // step_down) * self.base_channels * self.channel_multipliers[-1]
            self.final_linear = nn.Linear(dim, 1, bias=self.bias).to(input.device) 
        
        z = self.initial(input)
        z = self.mid(z)
        z = self.flatten(z)
        z = self.final_linear(z)

        return self.final_activation(z)        

class Discriminator2(nn.Module):
    def __init__(self, in_channels: int, channel_multipliers: list=[1], base_channels: int=16, bias: bool=False, **kwargs):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.bias = bias
        self.final_activation = nn.Identity() # Probability
        self.channel_multipliers = channel_multipliers

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels * channel_multipliers[0], 4, 2, 1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True)
        )
        layers = []
        for c, cp in zip(channel_multipliers, channel_multipliers[1:]):
            layers.append(nn.Conv2d(base_channels * c, base_channels * cp, 4, 2, 1, bias=bias))
            layers.append(nn.BatchNorm2d(base_channels * cp))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.mid = nn.Sequential(*layers)
        
        self.flatten = nn.Flatten()
        self.final_linear = None

    def forward(self, input):
        b, c, h, w = input.shape
        if self.final_linear is None:
            step_down = 2**len(self.channel_multipliers)
            dim = (h // step_down) * (w // step_down) * self.base_channels * self.channel_multipliers[-1]
            self.final_linear = nn.Linear(dim, 1, bias=self.bias).to(input.device) 
        
        z = self.initial(input)
        z = self.mid(z)
        z = self.flatten(z)
        z = self.final_linear(z)

        return self.final_activation(z)     
