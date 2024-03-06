import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from modules.util import get_norm, get_activation
from modules.embedding import PositionalEmbedding
from modules.misc.downsample import Downsample
from modules.misc.upsample import Upsample
from modules.residual_block import ResidualBlock, ResBlock
from modules.attention import AttentionBlock

class UNet(nn.Module):
    def __init__(self,
        in_channels,
        base_channels,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(),
        dropout=0.0,
        norm="gn",
        num_groups=32,
        time_emb_dim=None,
        num_classes=None,
        final_activation=None,
        activation="silu",
        **kwargs
        ):

        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.norm = norm
        self.num_groups = num_groups
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes
        self.activation = get_activation(activation)
        self.final_activation = nn.Identity() if final_activation is None else final_activation

        self.time_emb = nn.Sequential(
            PositionalEmbedding(self.base_channels, scale=1.0),
            nn.Linear(self.base_channels, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        ) if time_emb_dim is not None else None
        self.label_emb = nn.Embedding(self.num_classes, self.time_emb_dim) if num_classes is not None else None

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=self.activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions
                    )
                )
                now_channels = out_channels
                channels.append(now_channels)
            
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)

        self.mid = nn.ModuleList([
            ResBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=self.activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=True,
            ),
            ResBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=self.activation,
                norm=norm,
                num_groups=num_groups,
            ),
        ])

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=self.activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions
                    )
                )
                now_channels = out_channels
                
            if i != 0:
                self.ups.append(Upsample(now_channels))
        
        assert len(channels) == 0

        self.final = nn.Sequential(
            get_norm(norm, base_channels, num_groups),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            self.final_activation,
        )

    def forward(self, x, time=None, y=None):
        if self.time_emb is not None:
            if time is None:
                raise ValueError("time conditioning was specified but tim is not passed")
            time_emb = self.time_emb(time)
        else:
            time_emb = None
        if self.label_emb is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")
            label_emb = self.label_emb(y)  
        else:
            label_emb = None  

        x = self.init_conv(x)
        skips = [x]
        
        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)

        for layer in self.mid:
            x = layer(x, time_emb, y)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, y)
            
        x = self.final(x)
        return x