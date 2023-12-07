import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Encoder, Decoder, VectorQuantizer

class VQVAE(nn.Module):
    def __init__(self,
        in_channels,
        base_channels,
        emb_dim,
        num_embeddings=100,
        channel_multipliers=(1, 2, 4, 8),
        attention_resolutions=(),
        num_res_blocks=1,
        time_emb_dim=None,
        dropout=0.0,
        norm="gn",
        num_groups=16,
        **kwargs
        ):
        super().__init__()
        self.in_channels=in_channels
        self.base_channels=base_channels
        self.emb_dim=emb_dim
        self.num_embeddings=num_embeddings
        self.channel_multipliers=channel_multipliers
        self.num_res_blocks=num_res_blocks
        self.time_emb_dim=time_emb_dim
        self.dropout=dropout
        self.norm=norm
        self.num_groups=num_groups

        self.encoder = Encoder(in_channels, 
                               base_channels, 
                               emb_dim,
                               channel_multipliers=channel_multipliers,
                               attention_resolutions=attention_resolutions,
                               num_res_blocks=num_res_blocks,
                               time_emb_dim=time_emb_dim,
                               dropout=dropout,
                               norm=norm,
                               num_groups=num_groups,
                               )

        self.decoder = Decoder(in_channels, base_channels, emb_dim,
                               channel_multipliers=channel_multipliers,
                               attention_resolutions=attention_resolutions,
                               num_res_blocks=num_res_blocks,
                               time_emb_dim=time_emb_dim,
                               dropout=dropout,
                               norm=norm,
                               num_groups=num_groups,
                               )

        self.quantize = VectorQuantizer(num_embeddings, emb_dim)


    def encode(self, x, time_emb=None):
        z = self.encoder(x, time_emb=time_emb)
        quant, emb_loss = self.quantize(z)
        return quant, emb_loss


    def decode(self, z, time_emb=None):
        z = self.decoder(z, time_emb=time_emb)
        return z
    

    def forward(self, x, y=None, time_emb=None):
        z = self.encoder(x, time_emb=time_emb)
        quant, _ = self.quantize(z)
        x_r = self.decode(quant, time_emb=time_emb)
        return x_r
    
    def loss(self, x, time_emb=None, y=None, measure='l2', vq_weight=1):
        rec_loss = 0
        vq_loss = 0
        quant, vq_loss = self.encode(x, time_emb=time_emb)
        x_r = self.decode(quant, time_emb=time_emb)
        if measure=='l1':
            rec_loss = F.l1_loss(x, x_r)
        elif measure=='l2':
            rec_loss = F.mse_loss(x, x_r)
        loss = rec_loss + vq_weight * vq_loss
        return {'loss': loss, 'rec_loss': rec_loss, 'vq_loss': vq_loss}
    
    def training_step(self, x, time_emb=None,y=None, measure='l2', vq_weight=1):
        loss = self.loss(x,time_emb=time_emb,y=y,measure=measure,vq_weight=vq_weight)
        return loss['loss']