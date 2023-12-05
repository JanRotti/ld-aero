import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Encoder, Decoder

class VAE(nn.Module):
    def __init__(self,
        in_channels,
        base_channels,
        emb_dim,
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
        self.emb_dim = emb_dim

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
                               double_latent=True,
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



    def encode(self, x, time_emb=None):
        z = self.encoder(x, time_emb=time_emb)
        mu, log_var = torch.split(z, self.emb_dim, dim=1)
        return mu, log_var

    def decode(self, z, time_emb=None):
        z = self.decoder(z, time_emb=time_emb)
        return z
    
    def forward(self, x, time_emb=None):
        mu, log_var = self.encode(x, time_emb=time_emb)
        z = self.reparameterize(mu, log_var)
        x_r = self.decode(z, time_emb=time_emb)
        return x_r
    
    def loss(self, x, time_emb=None, y=None, measure='l2', kl_weight=0.0001):
        rec_loss = 0
        kl_loss = 0
        mu, log_var = self.encode(x, time_emb=time_emb)
        z = self.reparameterize(mu, log_var)
        x_r = self.decode(z, time_emb=time_emb)
        if measure=='l1':
            rec_loss = F.l1_loss(x, x_r)
        elif measure=='l2':
            rec_loss = F.mse_loss(x, x_r)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=(1,2,3)), dim = 0)
        loss = rec_loss + kl_weight * kl_loss
        return {'loss': loss, 'rec_loss': rec_loss, 'kl_loss': kl_loss}
    
    def training_step(self, x, time_emb=None,y=None, measure='l2', kl_weight=0.0001):
        loss = self.loss(x,time_emb=time_emb,y=y,measure=measure,kl_weight=kl_weight)
        return loss['loss']

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z