import torch
import torch.nn as nn
import torch.nn.functional as F

from ..residual_block import ResidualBlock
from ..misc import Upsample
from ..util import get_norm

class Decoder(nn.Module):
    
    def __init__(self,
        out_channels,
        base_channel,
        z_channels,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=1,
        attention_resolutions=(),
        time_emb_dim=None,
        dropout=0.0,
        norm="bn",
        num_groups=32,
        output_activation=nn.Sigmoid(),
        **kwargs
        ):

        super().__init__()

        self.base_channel = base_channel
        self.out_channel = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(channel_multipliers)
        self.channel_multipliers = (1,) + tuple(channel_multipliers)
        self.time_emb_dim = time_emb_dim
        self.output_activation = output_activation
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
                                out_channels, 
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
        h = self.output_activation(h)
        return h
