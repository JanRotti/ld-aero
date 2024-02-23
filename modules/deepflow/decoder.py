import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, latent_dim=None, output_dim=None, heads=None, head_size=None, macMHA=True, macMLP=True, attention_map=False, name=None, mlp_second_dense=True, dropout=0.0):
        super(MultiHeadAttentionBlock, self).__init__()
        self.latent_dim = latent_dim
        self.heads = heads
        self.macMHA = macMHA
        self.macMLP = macMLP
        self.create_attention_map = attention_map
        self.attention_map = None
        self.mlp_second_dense = mlp_second_dense
        if output_dim is None:
            output_dim = latent_dim
        if head_size is None:
            self.head_size = self.latent_dim // self.heads
        else:
            self.head_size = head_size
        self.mha = nn.MultiheadAttention(embed_dim=self.latent_dim, num_heads=self.heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(self.latent_dim)
        self.ln0 = nn.LayerNorm(self.latent_dim)
        self.mlp0 = MLPBlock(latent_dim=self.latent_dim, output_dim=output_dim, mac=self.macMLP, second_dense=self.mlp_second_dense)
        if self.macMHA:
            self.mac = MultiplyAddChannelsOneWeight()

    def forward(self, skip, qkv, mha_output=False):
        if self.create_attention_map:
            x, self.attention_map = self.mha(qkv, qkv, qkv)
        else:
            x, _ = self.mha(qkv, qkv, qkv)
        if self.macMHA:
            x = self.mac([skip, x])
        xmha = x
        x = self.mlp0(x)
        if mha_output:
            return x, self.attention_map, xmha
        else:
            return x, self.attention_map


class MLPBlock(nn.Module):
    def __init__(self, latent_dim=None, output_dim=None, name=None, mac=True, second_dense=True):
        super(MLPBlock, self).__init__()
        if output_dim is None:
            self.output_dim = latent_dim
        else:
            self.output_dim = output_dim
        self.second_dense = second_dense
        self.ln0 = nn.LayerNorm(latent_dim)
        self.dense0 = nn.Linear(latent_dim, latent_dim*2)
        if self.second_dense:
            self.dense1 = nn.Linear(latent_dim*2, self.output_dim)
        if mac:
            self.mac = MultiplyAddChannelsOneWeight()
        else:
            self.mac = None

    def forward(self, inputs):
        x = inputs
        x = self.dense0(x)
        x = F.gelu(x)
        if self.second_dense:
            x = self.dense1(x)
        if self.mac is not None:
            x = self.mac([x, inputs])
        return x


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
    def __init__(self, name=None, num_heads=8, dropout=0.3, filters=64, head_size=4, multipliers=None):
        super(BaseLayer, self).__init__()
        self.filters = filters
        self.multipliers = multipliers
        self.blocks = []
        self.num_heads = num_heads
        self.latent_dim = filters
        self.head_size = head_size
        for m in self.multipliers:
            tup = (MultiHeadAttentionBlock(latent_dim=self.latent_dim, output_dim=self.latent_dim, heads=self.num_heads, head_size=self.head_size,
                                           macMHA=True, macMLP=True, attention_map=False, mlp_second_dense=True,
                                           dropout=dropout), ResLayer(filters=self.filters, multiplier=m))
            self.blocks.append(tup)

    def forward(self, inputs):
        x = inputs
        for m in self.multipliers:
            x, _attn = self.blocks[m][0](x, (x, x, x))
            x = self.blocks[m][1](x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_shape=None, output_dim=None, biasOnOff=True, init=None,
                 num_heads=8, filters_start=64, latent_dim=64, dropout=0.3, head_size=4, multipliers=None, pixelshuffle_smoothing=False, scale_layers=0, database=None):
        super(Decoder, self).__init__()
        self.epoch = 0
        self.output_dim = output_dim
        self.input_dim = input_shape
        self.num_heads = num_heads
        self.head_size = head_size
        self.multpliers = multipliers
        self.pixelshuffle_smoothing = pixelshuffle_smoothing
        self.latent_dim = latent_dim
        self.database = database

        f = filters_start
        self.flatten = nn.Flatten()
        self.pxlshuffle = []
        self.pxlshuffle.append(PixelShuffle(scale=4, filter=f, pixelshuffle_smoothing=self.pixelshuffle_smoothing))
        self.c2d_0 = nn.Conv2d(input_shape[2], f, kernel_size=(3, 3), padding='same')
        self.c2d_concat = nn.Conv2d(f*2, f, kernel_size=(3, 3), padding='same')
        self.baselayer = BaseLayer(name="BaseLayer", num_heads=num_heads, head_size=self.head_size, dropout=dropout, multipliers=self.multpliers, filters=f)
        self.mhab0 = MultiHeadAttentionBlock(latent_dim=latent_dim, output_dim=1, heads=self.num_heads, head_size=self.head_size,
                                             macMHA=True, macMLP=True, attention_map=False, mlp_second_dense=True,
                                             dropout=dropout)
        for i in range(scale_layers-1):
            f = f // 2
            self.pxlshuffle.append(PixelShuffle(scale=2, filter=f))
        f = filters_start
        self.bn_out = nn.BatchNorm2d(f)
        self.conv2dOut0 = nn.Conv2d(f, output_dim[2], kernel_size=(3, 3), padding='same')

    def forward(self, inp, warmup=False):
        bs = inp.shape[0]
        freevars = inp.shape[1]
        f = 32
        repeat_num = int(np.log2(self.output_dim[0])) - 1
        mult = 2 ** (repeat_num - 1)
        curr_filters = f * mult
        scale_counter = 0
        layer_count = len(self.pxlshuffle)

        generator = inp
        generator = self.c2d_0(generator)
        generator = F.swish(generator)
        generator_skip = generator
        generator = self.baselayer(generator)
        generator = torch.cat((generator_skip, generator), dim=1)
        generator = self.c2d_concat(generator)
        for i in range(scale_counter, layer_count-1):
            generator = self.pxlshuffle[scale_counter](generator)
            generator = F.swish(generator)
            scale_counter += 1
        generator = self.bn_out(generator)
        generator = F.swish(generator)
        generator = self.conv2dOut0(generator)
        out = generator.view(bs, self.output_dim[0], self.output_dim[1], self.output_dim[2])
        return out
