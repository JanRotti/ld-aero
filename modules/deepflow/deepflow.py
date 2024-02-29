import torch
import torch.nn as nn
import torch.nn.functional as F
from ..util import get_norm
#############################################################
# Original
#############################################################
class MultiplyAddChannelsOneWeight(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, inputs):
        if len(inputs) != 2:
            raise Exception('An multiplyAddChannelsOneWeight layer should have 2 inputs')

        origin = inputs[0]
        t2 =  inputs[1]
        return torch.add(torch.multiply(t2, self.gamma), origin)

class Sampling(nn.Module):
    def forward(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = torch.randn(*z_mean.shape, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

"""
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, latent_dim, output_dim, head_size=4, macMHA=True, macMLP=True, create_attention_map=True, mlp_second_dense=True, **kwargs):
        super().__init__()
        self.macMHA = macMHA
        self.macMLP = macMLP

        self.attention_map = None
        self.create_attention_map = create_attention_map
        self.mlp_second_dense = mlp_second_dense

        self.head_size = head_size
        if output_dim is None:
            output_dim = latent_dim
        self.mha = F.scaled_dot_product_attention
        self.mlp0 = MLPBlock(latent_dim=self.latent_dim, output_dim=output_dim, mac=self.macMLP,second_dense=self.mlp_second_dense)
        if self.macMHA:
            self.mac = MultiplyAddChannelsOneWeight()

        self.qLinear = nn.Linear(latent_dim, self.head_size)
        self.kLinear = nn.Linear(latent_dim, self.head_size)
        self.vLinear = nn.Linear(latent_dim, self.head_size)
        self.qln = nn.LayerNorm((-1, self.head_size), epsilon=1e-6, elementwise_affine=True)
        self.kln = nn.LayerNorm((-1, self.head_size), epsilon=1e-6, elementwise_affine=True)
        self.vln = nn.LayerNorm((-1, self.head_size), epsilon=1e-6, elementwise_affine=True)

    def forward(self, skip, qkv, mha_output=False):

        q = self.qLinear(qkv[0])
        k = self.kLinear(qkv[1])
        v = self.vLinear(qkv[2])

        skip = self.qDense(skip)

        #q = self.qln(q)
        #k = self.kln(k)
        #v = self.vln(v)

        if self.create_attention_map:
            x, self.attention_map = self.mha(q, k, v)
        else:
            x = self.mha(q, k, v)
        if self.macMHA:
            x = self.mac([skip, x])

        xmha = x
        x = self.mlp0(x)

        if mha_output:
            return x, self.attention_map#, xmha
        else:
            return x, self.attention_map
"""
class MultiHeadAttentionBlock(nn.Module):
    __doc__ = r"""Applies multi-head QKV self-attention with a residual connection.
    
    Input:
        x: tensor of shape (N, in_channels, H, W)
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        num_heads (int): number of attention heads. Default: 4
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
    """
    def __init__(self, in_channels, norm="gn", num_groups=32, num_heads=4):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        self.num_heads = num_heads
        
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3 * num_heads, 1)
        self.to_out = nn.Conv2d(in_channels * num_heads, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(self.norm(x))
        qkv = qkv.view(b, self.num_heads, -1, h, w)
        q, k, v = torch.split(qkv, c, dim=2)

        q = q.permute(0, 1, 3, 4, 2).view(b * self.num_heads, h * w, c)
        k = k.view(b * self.num_heads, c, h * w)
        v = v.permute(0, 1, 3, 4, 2).view(b * self.num_heads, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b * self.num_heads, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b * self.num_heads, h * w, c)
        out = out.view(b, self.num_heads, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        out = out.view(b, self.num_heads * c, h, w)

        return self.to_out(out) + x

class MLPBlock(nn.Module):
    def __init__(self, latent_dim=None, output_dim=None, mac=True, second_dense=False, **kwargs):
        super().__init__(**kwargs)
        
        self.output_dim = latent_dim if output_dim == None else output_dim
        self.second_dense = second_dense    
        self.ln0 = nn.LayerNorm((-1, latent_dim), epsilon=1e-6, trainable=True)
        self.dense0 = nn.Linear(latent_dim, self.output_dim)
        if self.second_dense:
            self.dense1 = nn.Linear(self.output_dim, self.output_dim)
        if mac:
            self.mac = MultiplyAddChannelsOneWeight()
        else:
            self.mac=None
         
    def forward(self, inputs):
        x = inputs
           
        x = self.dense0(self.ln0(x))
        x = F.gelu(x, approximate='tanh')
        if self.second_dense:
            x = self.dense1(x)
        if self.mac is not None:
            x = self.mac([x,inputs])

        return x    

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, downsize=2,**kwargs):
        super().__init__(**kwargs)
        self.downsize = downsize
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.c2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), padding='same')
        self.c2_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding='same')
        self.c2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding='same')
        self.activation = nn.LeakyReLU()
        self.pool = nn.AvgPool2d(downsize, downsize) 

    def forward(self, inputs):
        x_skip = inputs
        x_skip = self.c2_1(x_skip)
        if self.downsize > 1:
            x_skip = self.pool(x_skip)
        x = inputs
        x = self.c2_2(x)
        x = self.activation(x)
        x = self.c2_3(x)
        if self.downsize > 1:
            x = self.pool(x)
        x = x + x_skip
        return x

class Encoder(nn.Module):
    def __init__(self, input_shape, base_channels=16, latent_dim=128, image_layers=1, **kwargs):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_dim = input_shape
        self.image_layers = image_layers
        self.base_channels = base_channels

        resblocks=[]
        resblocks.append(
                ResLayer(in_channels=self.input_dim[0], out_channels=self.base_channels, downsize=2)
        )
        for i in range(1,self.image_layers):
            resblocks.append(
                ResLayer(in_channels=self.base_channels*(i), out_channels=self.base_channels*(i+1),downsize=2)
            )
        self.mid = nn.Sequential(*resblocks)    
        
        self.res_block_same = ResLayer(in_channels=self.base_channels*self.image_layers,out_channels=self.latent_dim,downsize=1)
        
        wr, hr = (int(self.input_dim[1] / 2**self.image_layers), int(self.input_dim[2] / 2**self.image_layers))
        self.flatten = nn.Flatten(start_dim=1)
        self.z_mean = nn.Linear(wr*hr*self.latent_dim, self.latent_dim)
        self.z_var = nn.Linear(wr*hr*self.latent_dim, self.latent_dim)

        self.sampling = Sampling()

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        z = inputs
        z = self.mid(z)
        z = self.res_block_same(z)
        z = self.flatten(z)
        z_mean = self.z_mean(z)
        z_var = self.z_var(z)
        z = self.sampling([z_mean, z_var])
        return z_mean, z_var, z


#############################################################
# Decoder with Adaption
#############################################################

class BaseLayer(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_heads=8, n_layers=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.mid = nn.Sequential(
            MultiHeadAttentionBlock(in_channels=self.in_channels, num_heads=self.num_heads),
            ResLayer(in_channels=self.in_channels, out_channels=self.in_channels, downsize=0)
        )
        layers = []
        for i in range(1,n_layers):
            layers.append(MultiHeadAttentionBlock(in_channels=self.in_channels, num_heads=self.num_heads))
            layers.append(ResLayer(in_channels=self.in_channels, out_channels=self.in_channels, downsize=0))
        self.block = nn.Sequential(*layers) if len(layers) > 1 else nn.Identity()
        self.final = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3,3), padding='same')

    def forward(self, inputs):
        x = inputs
        x = self.mid(x)
        x = self.block(x)
        x = self.final(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_shape=None, num_heads=8, in_channels=64, latent_dim=64, image_layers=1, **kwargs):
        super().__init__()

        upscale_factor = 2**image_layers
        wr, hr = input_shape[1] // upscale_factor, input_shape[2] // upscale_factor
        self.reduced_dim = (wr, hr)
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.latent_dim = latent_dim

        self.ini = nn.Linear(self.latent_dim, wr*hr*self.latent_dim)
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
        b, l = input.shape
        wr, hr = self.reduced_dim
        assert l == self.latent_dim
        x = self.ini(input).view(-1, self.latent_dim, wr, hr)
        x = self.c2d_0(x)
        x = self.silu(x)
        x_skip = x
        x = self.baselayer(x)
        x = torch.cat((x_skip, x), dim=1)
        x = self.c2d_concat(x)

        x = self.upscale(x)
        x = self.bn_out(x)
        x = self.silu(x)
        x = self.conv2dOut0(x)

        return self.sigmoid(x)