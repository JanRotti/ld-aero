import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MultiplyAddChannelsOneWeight, MLPBlock

# TODO: look into deepflow_vae_decoder.py and deepflow_vqvae_encoder
class ResLayer(nn.Module):
    def __init__(self, name=None, filters=None, multiplier=4):
        super(ResLayer, self).__init__()
        self.filters = filters
        self.multiplier = multiplier
        self.c2_1 = nn.Conv2d(filters*self.multiplier, filters*self.multiplier, kernel_size=(1,1), padding='same')
        self.c2_2 = nn.Conv2d(filters*self.multiplier, filters*self.multiplier, kernel_size=(3,3), padding='same')
        self.c2_3 = nn.Conv2d(filters*self.multiplier, filters, kernel_size=(1,1), padding='same')

    def forward(self, inputs):
        x_skip = inputs
        x = inputs
        x = self.c2_1(x)
        x = F.swish(x)
        x = self.c2_2(x)
        x = F.swish(x)
        x = self.c2_3(x) + x_skip
        return x

class BaseLayer(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_heads=8, n_layers=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.mid = nn.Sequential(
            MultiHeadAttentionBlock(in_channels=self.in_channels, num_heads=self.num_heads),
            ResBlock(in_channels=self.in_channels, out_channels=self.in_channels, downsample=False)
        )
        layers = []
        for i in range(1,n_layers):
            layers.append(MultiHeadAttentionBlock(in_channels=self.in_channels, num_heads=self.num_heads))
            layers.append(ResBlock(in_channels=self.in_channels, out_channels=self.in_channels, downsample=False))
        self.block = nn.Sequential(*layers) if len(layers) > 1 else nn.Identity()
        self.final = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3,3), padding='same')

    def forward(self, inputs):
        x = inputs
        x = self.mid(x)
        x = self.block(x)
        x = self.final(x)
        return x

# GO to config_model_description and deepflow_vae_decoder
class Decoder(nn.Module):
    def __init__(self, input_shape=None, num_heads=8, in_channels=64, latent_dim=64, image_layers=1, **kwargs):
        super().__init__()
        upscale_factor = 2**image_layers
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.latent_dim = latent_dim

        self.pxlshuffle = nn.PixelShuffle(2)
        self.silu = torch.nn.SiLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.bn_out = nn.BatchNorm2d(in_channels)

        self.c2d_0 = nn.Conv2d(self.latent_dim, in_channels, kernel_size=(3, 3), padding='same')
        self.c2d_concat = nn.Conv2d(in_channels*2, in_channels, kernel_size=(3, 3), padding='same')
        self.baselayer = BaseLayer(in_channels=in_channels, out_channels=in_channels, num_heads=num_heads, n_layers=1)

        layers = []
        for i in range(upscale_factor//2):
            layers.append(nn.Conv2d(in_channels, in_channels*4,kernel_size=(3, 3), padding='same'))
            layers.append(self.pxlshuffle)
            layers.append(self.silu)

        self.upscale = nn.Sequential(*layers)
        self.conv2dOut0 = nn.Conv2d(in_channels, self.input_shape[0], kernel_size=(3, 3), padding='same')


    def forward(self, input):
        b, l, wr, hr = input.shape
        assert l == self.latent_dim

        generator = self.c2d_0(input)
        generator = self.silu(generator)
        generator_skip = generator
        generator = self.baselayer(generator)
        generator = torch.cat((generator_skip, generator), dim=1)
        generator = self.c2d_concat(generator)

        generator = self.upscale(generator)
        generator = self.bn_out(generator)
        generator = self.silu(generator)
        generator = self.conv2dOut0(generator)
    
        return self.sigmoid(generator)



#############################################################
# Original
#############################################################
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, latent_dim, output_dim=None, heads=None, head_size=None, macMHA=True, macMLP=True, attention_map=False, mlp_second_dense=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.latent_dim = latent_dim
        self.heads = heads
        self.macMHA = macMHA
        self.macMLP = macMLP
        self.create_attention_map = attention_map
        self.attention_map = None
        self.mlp_second_dense = mlp_second_dense

        if output_dim is None:
            output_dim = latent_dim

        if head_size==None:
            assert(heads is not None, "Either 'head_size' or 'heads' has to be defined.")
            self.head_size = self.latent_dim // self.heads
        else:
            self.head_size = head_size

        self.mha = nn.MultiheadAttention(head_size=self.head_size, num_heads=self.heads,
                                                 return_attn_coef=self.create_attention_map)
        #self.ln1 = nn.LayerNormalization((-1, self.head_size), epsilon=1e-6, trainable=True)
        #self.ln0 = nn.LayerNormalization((-1, self.head_size), epsilon=1e-6, trainable=True)
        self.mlp0 = MLPBlock(latent_dim=self.latent_dim, output_dim=output_dim, mac=self.macMLP, second_dense=self.mlp_second_dense)
        if self.macMHA:
            self.mac = MultiplyAddChannelsOneWeight()


    def forward(self, skip, qkv, mha_output=False):

        if self.create_attention_map:
            x, self.attention_map = self.mha(qkv)  
        else:
            x, _ = self.mha(qkv)#
        if self.macMHA:
            x = self.mac([skip, x])

        #x = self.ln1(x)
        xmha = x
        x = self.mlp0(x)

        if mha_output:
            return x, self.attention_map, xmha
        else:
            return x, self.attention_map

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, downsize=0,**kwargs):
        super().__init__(**kwargs)
        self.downsize = downsize
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.c2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), padding='same')
        self.c2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding='same')
        self.c2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding='same')
        self.activation = nn.SiLU()
        self.pool = nn.AveragePool2d(downsize, downsize) 

    def forward(self, inputs):
        x_skip = inputs
        x_skip = self.c2_1(x_skip)
        if self.downsize > 1:
            x_skip = self.pool(x_skip)
        x = inputs
        x = self.c2_2(x)
        x = self.activation(x)
        x = self.c2_3(x)
        x = self.activation(x)
        if self.downsize > 1:
            x = self.pool(x)
        x = x + x_skip
        return x