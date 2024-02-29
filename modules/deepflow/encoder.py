import torch
import torch.nn as nn
import torch.nn.functional as F

#from .decoder import MultiHeadAttentionBlock
#from .res_layer import ResidualLayer
#from ..distribution import DiagonalGaussianDistribution

"""
class Encoder(nn.Module):

    def __init__(self, 
            input_shape=None,
            base_channels=16, 
            latent_dim=128, 
            image_layers=1,
            **kwargs):

        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_shape
        self.image_layers = image_layers
        self.base_channels = base_channels

        resblocks=[]
        resblocks.append(
            nn.Sequential(
                ResidualLayer(channels=self.input_dim[0], downsize=True),
                nn.Conv2d(self.input_dim[0], self.base_channels, kernel_size=(3,3), padding='same')
                )
            )
        for i in range(1,self.image_layers):
            resblocks.append(
                nn.Sequential(
                    ResidualLayer(channels=self.base_channels*(i), downsize=True),
                    nn.Conv2d(self.base_channels*(i), self.base_channels*(i+1), kernel_size=(3,3), padding='same')   
                )
            )
        self.mid = nn.Sequential(*resblocks)    
        self.conv = nn.Conv2d(self.base_channels * self.image_layers, self.latent_dim, kernel_size=(3,3), padding='same')  
        self.res_block_same = ResidualLayer(channels=self.latent_dim)
        
        wr, hr = (int(self.input_dim[1] / 2**self.image_layers), int(self.input_dim[2] / 2**self.image_layers))
        self.flatten = nn.Flatten(start_dim=1)# b c (h*w)
        
        self.z_mean = nn.Linear(wr*hr*self.latent_dim, wr*hr*self.latent_dim)
        self.z_var  = nn.Linear(wr*hr*self.latent_dim, wr*hr*self.latent_dim)
        self.reduced_dim = (self.latent_dim, wr, hr)

    def forward(self, inputs):
        b, c, w, h = inputs.shape
        z = inputs

        z = self.mid(z)

        z = self.conv(z)
        z = self.res_block_same(z)

        z = self.flatten(z)
        z_mean = self.z_mean(z)
        z_var = self.z_var(z)
        z_mean = torch.reshape(z_mean, (b, *self.reduced_dim))
        z_var = torch.reshape(z_var, (b, *self.reduced_dim))
        epsilon = torch.randn(z_mean.shape, device=z_mean.device)
        return z_mean, z_var, z_mean + torch.exp(0.5 * z_var) * epsilon
"""

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
    def foward(self, inputs):
        z_mean, z_log_var = inputs
        b, c, w, h = z_mean.shape
        epsilon = torch.randn(b, c, w, h, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

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


class MLPBlock(nn.module):
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
        self.c2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding='same')
        self.c2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding='same')
        self.activation = nn.LeakyReLU()
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
                ResLayer(in_channels=self.input_dim[0], out_channels=self.base_channels, downsize=True)
        )
        for i in range(1,self.image_layers):
            resblocks.append(
                ResLayer(in_channels=self.base_channels*(i), out_channels=self.base_channels*(i+1),downsize=True)
            )
        self.mid = nn.Sequential(*resblocks)    
        
        self.res_block_same = ResLayer(in_channels=self.base_channels*self._image_layers,out_channels=self.latent_dim)
        
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


