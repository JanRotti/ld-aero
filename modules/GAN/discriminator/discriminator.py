import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels: int, channel_multipliers: list=[1], base_channels: int=16, discr_activation=nn.Sigmoid(), bias: bool=False, **kwargs):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.bias = bias
        self.final_activation = discr_activation
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

class ClassifyingDiscriminator(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, channel_multipliers: list=[1], base_channels: int=16, discr_activation=nn.Sigmoid(), bias: bool=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.bias = bias
        self.final_activation = discr_activation
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
        self.classification = None
        self.decision = None

    def forward(self, input):
        b, c, h, w = input.shape
        if self.decision is None or self.classification is None:
            step_down = 2**len(self.channel_multipliers)
            dim = (h // step_down) * (w // step_down) * self.base_channels * self.channel_multipliers[-1]
            self.decision = nn.Linear(dim, 1, bias=self.bias).to(input.device) 
            self.classification = nn.Linear(dim, self.n_classes, bias=self.bias).to(input.device)

        z = self.initial(input)
        z = self.mid(z)
        z = self.flatten(z)
        c = self.classification(z)
        d = self.decision(z)

        return self.final_activation(d), c        


class InfoGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, code_dim: int, channel_multipliers: list=[1], base_channels: int=16, discr_activation=nn.Sigmoid(), bias: bool=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.code_dim = code_dim
        self.base_channels = base_channels
        self.bias = bias
        self.final_activation = discr_activation
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
        self.classification = None
        self.latent_layer = None
        self.decision = None

    def forward(self, input):
        b, c, h, w = input.shape
        if self.decision is None or self.classification is None or self.latent_layer is None:
            step_down = 2**len(self.channel_multipliers)
            dim = (h // step_down) * (w // step_down) * self.base_channels * self.channel_multipliers[-1]
            self.decision = nn.Linear(dim, 1, bias=self.bias).to(input.device) 
            self.classification = nn.Linear(dim, self.n_classes, bias=self.bias).to(input.device)
            self.latent_layer = nn.Linear(dim, self.code_dim, bias=self.bias).to(input.device)

        z = self.initial(input)
        z = self.mid(z)
        z = self.flatten(z)
        c = self.classification(z)
        d = self.decision(z)
        l = self.latent_layer(z)
        return self.final_activation(d), c, l        