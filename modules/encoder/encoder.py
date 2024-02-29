import torch
import torch.nn as nn
import torch.nn.functional as F

from ..residual_block import ResidualBlock
from ..misc import Downsample
from ..util import get_norm

class Encoder(nn.Module):
    def __init__(self, 
        in_channel,
        base_channel,
        z_channels,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=1,
        time_emb_dim=None,
        dropout=0.0,
        norm="bn",
        num_groups=32,
        attention_resolutions=(),
        double_z=False,
        **ignore_kwargs
        ):

        super().__init__()

        self.in_channel = in_channel
        self.base_channel = base_channel
        self.z_channels = z_channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(channel_multipliers)
        self.channel_multipliers = (1,) + tuple(channel_multipliers)
        self.time_emb_dim = time_emb_dim

        # Initial
        self.conv_in = nn.Conv2d(in_channel, 
            base_channel, 
            kernel_size=3, stride=1, padding=1
            )

        # Downsampling
        self.down = nn.ModuleList()
        for i in range(self.num_resolutions):
            res_blocks = nn.ModuleList()
            block_in  = base_channel * self.channel_multipliers[i]
            block_out = base_channel * channel_multipliers[i]
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
                block.downsample = Downsample(block_out)
            self.down.append(block)

        block_in = self.channel_multipliers[-1] * base_channel
        
        # middle
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

        # end
        self.norm_out = get_norm(norm, block_in, num_groups=num_groups)
        self.conv_out = nn.Conv2d(block_in, 
                                2 * z_channels if double_z else z_channels, 
                                kernel_size=3,
                                stride=1,
                                padding=1
        )

    def forward(self, x, time_emb=None):
        h = self.conv_in(x)
        # down
        for i in range(self.num_resolutions):
            block = self.down[i]
            for j in range(self.num_res_blocks):
                h = block.block[j](h, time_emb=time_emb)
            if i != self.num_resolutions-1:
                h = self.down[i].downsample(h)
        # middle
        h = self.mid.block_1(h, time_emb=time_emb)
        h = self.mid.block_2(h, time_emb=time_emb)
        # final
        h = self.norm_out(h)
        h = self.conv_out(h)
        return h

