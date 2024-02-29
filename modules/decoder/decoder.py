import torch
import torch.nn as nn
import torch.nn.functional as F

from ..residual_block import ResidualBlock
from ..misc import Upsample
from ..util import get_norm
from ..attention import MultiHeadAttentionBlock

class Decoder(nn.Module):
    
    def __init__(self,
        in_channel,
        base_channel,
        z_channels,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=1,
        attention_resolutions=(),
        time_emb_dim=None,
        dropout=0.0,
        norm="bn",
        num_groups=32,
        final_activation=nn.Sigmoid(),
        **kwargs
        ):

        super().__init__()

        self.base_channel = base_channel
        self.out_channel = in_channel
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(channel_multipliers)
        self.channel_multipliers = (1,) + tuple(channel_multipliers)
        self.time_emb_dim = time_emb_dim
        self.final_activation = final_activation
        # 
        block_in = base_channel * self.channel_multipliers[-1]

        # Initial
        self.conv_in = nn.Conv2d(z_channels, 
            block_in, 
            kernel_size=3, stride=1, padding=1
            )
        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResidualBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       time_emb_dim=time_emb_dim,
                                       dropout=dropout,
                                       norm=None,
                                       num_groups=num_groups,
                                       use_attention=True)
        self.mid.block_2 = ResidualBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       time_emb_dim=time_emb_dim,
                                       dropout=dropout,
                                       norm=None,
                                       num_groups=num_groups,
                                       use_attention=True)
        # Upsampling
        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            res_blocks = nn.ModuleList()
            block_out = base_channel * self.channel_multipliers[i]
            for j in range(self.num_res_blocks):
                res_blocks.append(ResidualBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         time_emb_dim=self.time_emb_dim,
                                         dropout=dropout,
                                         norm=norm,
                                         num_groups=num_groups,
                                         use_attention=i in attention_resolutions))
                block_in = block_out
            block = nn.Module()
            block.block = res_blocks
            if i != self.num_resolutions-1:
                block.upsample = Upsample(block_in)
            self.up.insert(0, block)

        # end
        self.norm_out = get_norm(norm, block_in, num_groups=num_groups)
        self.conv_out = nn.Conv2d(block_in, 
                                in_channel, 
                                kernel_size=3,
                                stride=1,
                                padding=1
        )

    def forward(self, x, time_emb=None):
        h = self.conv_in(x)
        # middle
        h = self.mid.block_1(h, time_emb=time_emb)
        h = self.mid.block_2(h, time_emb=time_emb)
        # down
        for i in reversed(range(self.num_resolutions)):
            block = self.up[i]
            for j in range(self.num_res_blocks):
                h = block.block[j](h, time_emb=time_emb)
            if i != self.num_resolutions-1:
                h = self.up[i].upsample(h)
        # final
        h = self.norm_out(h)
        h = self.conv_out(h)
        h = self.final_activation(h)
        return h



class AttentionDecoder(nn.Module):
    def __init__(self,
        in_channel,
        base_channel,
        z_channels,
        num_attention_blocks=1,
        channel_multipliers=(1, 2, 4, 8),
        attention_resolutions=(),
        num_heads=4,
        dropout=0.0,
        final_activation=nn.Sigmoid(),
        **kwargs
        ):

        super().__init__()
        self.in_channel = in_channel
        self.base_channel = base_channel
        self.z_channels = z_channels
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_attention_blocks = num_attention_blocks
        self.channel_multipliers = channel_multipliers
        
        self.upscale = 2**(len(channel_multipliers)-1)
        required_channels = 4 * in_channel #in_channel * self.upscale**2

        self.final_activation = final_activation
        self.activation = torch.nn.SiLU()

        layers = []
        for i in range(self.num_attention_blocks):
            layers.append(MultiHeadAttentionBlock(self.z_channels, num_heads=self.num_heads))
            layers.append(ResidualBlock(in_channels=self.z_channels, out_channels=self.z_channels,dropout=self.dropout))
        self.initial = nn.Sequential(*layers)
        layers = []
        channels = self.z_channels
        for i in range(1, (self.upscale // 2)-1):
            layers.append(ResidualBlock(in_channels=channels, out_channels=4*channels,dropout=self.dropout))
            layers.append(nn.PixelShuffle(2))
        self.mid = nn.Sequential(*layers)
        self.final = nn.Sequential(
            ResidualBlock(in_channels=channels, out_channels=required_channels,dropout=self.dropout),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.mid(x)
        x = self.final(x)
        x = self.final_activation(x)
        return x


class PSDecoder(nn.Module):
    
    def __init__(self,
        in_channel,
        base_channel,
        z_channels,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=1,
        attention_resolutions=(),
        time_emb_dim=None,
        dropout=0.0,
        norm="bn",
        num_groups=32,
        final_activation=nn.Sigmoid(),
        **kwargs
        ):

        super().__init__()

        self.base_channel = base_channel
        self.out_channel = in_channel
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(channel_multipliers)
        self.channel_multipliers = (1,) + tuple(channel_multipliers)
        self.time_emb_dim = time_emb_dim
        self.final_activation = final_activation
        # 
        block_in = base_channel * self.channel_multipliers[-1]

        # Initial
        self.conv_in = nn.Conv2d(z_channels, 
            block_in, 
            kernel_size=3, stride=1, padding=1
            )
        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResidualBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       time_emb_dim=time_emb_dim,
                                       dropout=dropout,
                                       norm=None,
                                       num_groups=num_groups,
                                       use_attention=True)
        self.mid.block_2 = ResidualBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       time_emb_dim=time_emb_dim,
                                       dropout=dropout,
                                       norm=None,
                                       num_groups=num_groups,
                                       use_attention=True)
        # Upsampling
        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            res_blocks = nn.ModuleList()
            block_out = base_channel * self.channel_multipliers[i]
            for j in range(self.num_res_blocks):
                res_blocks.append(ResidualBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         time_emb_dim=self.time_emb_dim,
                                         dropout=dropout,
                                         norm=norm,
                                         num_groups=num_groups,
                                         use_attention=i in attention_resolutions))
                block_in = block_out
            block = nn.Module()
            block.block = res_blocks
            if i != self.num_resolutions-1:
                block.upsample = nn.Sequential(
                    nn.Conv2d(block_in, block_out*4, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(2)
                )
            self.up.insert(0, block)

        # end
        self.norm_out = get_norm(norm, block_in, num_groups=num_groups)
        self.conv_out = nn.Conv2d(block_in, 
                                in_channel, 
                                kernel_size=3,
                                stride=1,
                                padding=1
        )

    def forward(self, x, time_emb=None):
        h = self.conv_in(x)
        # middle
        h = self.mid.block_1(h, time_emb=time_emb)
        h = self.mid.block_2(h, time_emb=time_emb)
        # down
        for i in reversed(range(self.num_resolutions)):
            block = self.up[i]
            for j in range(self.num_res_blocks):
                h = block.block[j](h, time_emb=time_emb)
            if i != self.num_resolutions-1:
                h = self.up[i].upsample(h)
        # final
        h = self.norm_out(h)
        h = self.conv_out(h)
        h = self.final_activation(h)
        return h