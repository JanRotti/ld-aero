import lightning.pytorch as pl
import torch
import torch.nn as nn
import numpy as np

from .base import Autoencoder

#from modules.deepflow.encoder import Encoder
#from modules.deepflow.decoder import Decoder
from modules.deepflow.deepflow import Encoder, Decoder
from modules.embedding.vector_quantizer import VectorQuantizer2 as VectorQuantizer

class DeepFlowVQVAE(Autoencoder):

    def __init__(self, config, latent_dim: int, n_embeddings: int=100, image_key="image", learning_rate: float=0.002, betas=(0.9, 0.99), kl_weight=0.0001):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.betas = betas
        self.kl_weight = kl_weight
        config["latent_dim"] = latent_dim
        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)
        self.q_layer = VectorQuantizer(n_embeddings, latent_dim)
        self.latent_dim = latent_dim
        self.image_key = image_key

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.q_layer.parameters()),
                                  lr=self.learning_rate, betas=self.betas)
        return opt_ae

    def training_step(self, data):
        mse = nn.MSELoss()
        real_fields = self.get_input(data, self.image_key)
        real_fields = real_fields.to(self.device)
        lossDict = {}
        
        z_mean, z_logvar, z = self.encoder(real_fields)
        reconstruct = self.decoder(z)
        
        reconstruct_loss = mse(reconstruct, real_fields)
        kl_loss = -0.5 * (1.0 + z_logvar - torch.square(z_mean) - torch.exp(z_logvar))
        kl_loss = torch.sum(kl_loss, axis=[1]) 
        kl_loss = self.kl_weight * torch.mean(kl_loss)
        loss = kl_loss + reconstruct_loss

        lossDict["vae_loss_kl"] = kl_loss.detach()
        lossDict["vae_loss_pix"] = reconstruct_loss.detach()
        lossDict["loss"] = loss.clone().detach()
        self.log_dict(lossDict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, data):
        mse = nn.MSELoss()
        real_fields = self.get_input(data, self.image_key)
        real_fields = real_fields.to(self.device)
        lossDict = {}

        z_mean, z_logvar, z = self.encoder(real_fields)
        reconstruct = self.decoder(z)
        
        reconstruct_loss = mse(reconstruct, real_fields)
        kl_loss = -0.5 *(1.0 + z_logvar - torch.square(z_mean) - torch.exp(z_logvar))
        kl_loss = torch.sum( kl_loss, axis=1 ) 
        kl_loss = self.kl_weight * torch.mean(kl_loss)
        
        loss = kl_loss + reconstruct_loss

        lossDict["vae_loss_kl"] = kl_loss.detach()
        lossDict["vae_loss_pix_me"] = nn.L1Loss()(reconstruct, real_fields).detach()
        lossDict["vae_loss_pix_mse"] = reconstruct_loss.detach()
        lossDict["loss"] = loss.clone().detach()
        self.log_dict(lossDict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss

    def forward(self, input):
        z_mean, z_logvar, z = self.encoder(input)
        reconstruct = self.decoder(z)
        return reconstruct  

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            z_mean, z_logvar, z = self.encoder(x)
            xrec = self.decoder(z)
            samples = self.decoder(torch.randn_like(z))
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
                samples = self.to_rgb(samples)
            log["samples"] = samples
            log["reconstructions"] = xrec
        
        log["inputs"] = x
        return log