import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels, image_size, latent_dim: int = 100, channel_multipliers: list=[1], base_channels: int=16, bias: bool=False, final_activation=nn.Sigmoid(), overexpand=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.channel_multipliers.reverse()
        self.bias = bias
        self.image_size = image_size # target
        self.final_activation = final_activation

        self.reduced_size = tuple([x // 2**len(channel_multipliers) for x in image_size])
        self.initial = nn.Linear(latent_dim, self.reduced_size[0] * self.reduced_size[1] * base_channels * self.channel_multipliers[0])
        layers = []
        for c, cp in zip(self.channel_multipliers, self.channel_multipliers[1:]):
            layers.append(nn.ConvTranspose2d(base_channels * c, base_channels * cp, 4, 2, 1, bias=bias))
            layers.append(nn.BatchNorm2d(base_channels * cp))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.mid = nn.Sequential(*layers)
        if overexpand:
            #self.final = nn.Sequential(
            #    nn.ConvTranspose2d(base_channels * channel_multipliers[-1], base_channels * channel_multipliers[-1], 4, 2, 1, bias=bias),
            #    nn.BatchNorm2d(base_channels * channel_multipliers[-1]),
            #    nn.LeakyReLU(0.2, inplace=True),
            #    nn.ConvTranspose2d(base_channels * channel_multipliers[-1], base_channels * channel_multipliers[-1], 4, 2, 1, bias=bias),
            #    nn.LeakyReLU(0.2, inplace=True),
            #    nn.Conv2d(base_channels * channel_multipliers[-1], self.in_channels, 4, 2, 1, bias=bias)
            #)
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(base_channels * channel_multipliers[-1], base_channels * channel_multipliers[-1], 3, 1, 1, bias=bias),
                nn.BatchNorm2d(base_channels * channel_multipliers[-1]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(base_channels * channel_multipliers[-1], base_channels * channel_multipliers[-1], 3, 1, 1, bias=bias),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * channel_multipliers[-1], self.in_channels, 4, 2, 1, bias=bias)
            )

        else:
            #self.final = nn.ConvTranspose2d(base_channels * channel_multipliers[-1], self.in_channels, 4, 2, 1, bias=bias)
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(base_channels * channel_multipliers[-1], self.in_channels, 3, 1, 1, bias=bias),
            )
        self.final_activation = final_activation

    def forward(self, input):
        z = self.initial(input)
        z = z.view(-1, self.base_channels * self.channel_multipliers[0], *self.reduced_size)
        z = self.mid(z)
        z = self.final(z)
        return self.final_activation(z)


class ConditionedGenerator(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, image_size, latent_dim: int = 100, channel_multipliers: list=[1], base_channels: int=16, bias: bool=False, final_activation=nn.Sigmoid(), overexpand=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.channel_multipliers.reverse()
        self.bias = bias
        self.image_size = image_size 
        self.final_activation = final_activation
        
        self.reduced_size = tuple([x // 2**len(channel_multipliers) for x in image_size])
        self.initial = nn.Linear(latent_dim + n_classes, self.reduced_size[0] * self.reduced_size[1] * base_channels * self.channel_multipliers[0])
        layers = []
        for c, cp in zip(self.channel_multipliers, self.channel_multipliers[1:]):
            layers.append(nn.ConvTranspose2d(base_channels * c, base_channels * cp, 4, 2, 1, bias=bias))
            layers.append(nn.BatchNorm2d(base_channels * cp))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.mid = nn.Sequential(*layers)
        if overexpand:
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(base_channels * channel_multipliers[-1], base_channels * channel_multipliers[-1], 3, 1, 1, bias=bias),
                nn.BatchNorm2d(base_channels * channel_multipliers[-1]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(base_channels * channel_multipliers[-1], base_channels * channel_multipliers[-1], 3, 1, 1, bias=bias),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * channel_multipliers[-1], self.in_channels, 4, 2, 1, bias=bias)
            )
        else:
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(base_channels * channel_multipliers[-1], self.in_channels, 3, 1, 1, bias=bias),
            )
        self.final_activation = final_activation

    def forward(self, input, labels):
        input = torch.cat((input, labels), dim=1)
        z = self.initial(input)
        z = z.view(-1, self.base_channels * self.channel_multipliers[0], *self.reduced_size)
        z = self.mid(z)
        z = self.final(z)
        return self.final_activation(z)


class InfoGANGenerator(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, code_dim: int, image_size, latent_dim: int = 100, channel_multipliers: list=[1], base_channels: int=16, bias: bool=False, final_activation=nn.Sigmoid(), overexpand=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.channel_multipliers.reverse()
        self.bias = bias
        self.image_size = image_size 
        self.final_activation = final_activation
        
        self.reduced_size = tuple([x // 2**len(channel_multipliers) for x in image_size])
        self.initial = nn.Linear(latent_dim + n_classes + code_dim, self.reduced_size[0] * self.reduced_size[1] * base_channels * self.channel_multipliers[0])
        layers = []
        for c, cp in zip(self.channel_multipliers, self.channel_multipliers[1:]):
            layers.append(nn.ConvTranspose2d(base_channels * c, base_channels * cp, 4, 2, 1, bias=bias))
            layers.append(nn.BatchNorm2d(base_channels * cp))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.mid = nn.Sequential(*layers)
        if overexpand:
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(base_channels * channel_multipliers[-1], base_channels * channel_multipliers[-1], 3, 1, 1, bias=bias),
                nn.BatchNorm2d(base_channels * channel_multipliers[-1]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(base_channels * channel_multipliers[-1], base_channels * channel_multipliers[-1], 3, 1, 1, bias=bias),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * channel_multipliers[-1], self.in_channels, 4, 2, 1, bias=bias)
            )
        else:
            self.final = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(base_channels * channel_multipliers[-1], self.in_channels, 3, 1, 1, bias=bias),
            )
        self.final_activation = final_activation

    def forward(self, input, labels, code):
        input = torch.cat((input, labels, code), dim=1)
        z = self.initial(input)
        z = z.view(-1, self.base_channels * self.channel_multipliers[0], *self.reduced_size)
        z = self.mid(z)
        z = self.final(z)
        return self.final_activation(z)