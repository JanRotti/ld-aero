import lightning.pytorch as pl
import torch
import torch.nn as nn
import numpy as np

from .base import Autoencoder

from modules.deepflow.encoder import Encoder
from modules.deepflow.decoder import Decoder
from modules.embedding.vector_quantizer import VectorQuantizer2 as VectorQuantizer

class DeepFlowVQVAE(Autoencoder):

    def __init__(self, config, latent_dim, learning_rate=0.002, betas=(0.9, 0.99)):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.betas = betas
        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)
        self.q_layer = VectorQuantizer(64, latent_dim)
        self.latent_dim = latent_dim

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.q_layer.parameters()),
                                  lr=self.learning_rate, betas=self.betas)
        return opt_ae

    def training_step(self, data):
        mse = nn.MSELoss()
        real_fields = self.get_input(data, "image")

        lossDict = {}
        
        z_mean, z_logvar, z = self.encoder(real_fields)
        reconstruct = self.decoder(z)
        
        reconstruct_loss = mse(reconstruct, real_fields)
        kl_loss = -0.5 * (1.0 + z_logvar - torch.square(z_mean) - torch.exp(z_logvar))
        kl_loss = torch.sum(kl_loss, axis=[1, 2, 3]) 
        kl_loss = torch.mean(kl_loss)
        loss = kl_loss + reconstruct_loss

        lossDict["vae_loss_kl"] = kl_loss
        lossDict["vae_loss_pix"] = reconstruct_loss
        lossDict["loss"] = loss
        
        return lossDict

    def validation_step(self, data):
        mse = nn.MSELoss()
        real_fields = self.get_input(data, "image")

        lossDict = {}

        z_mean,z_logvar,z = self.encoder(real_fields)
        reconstruct = self.decoder(z)
        
        reconstruct_loss = mse(reconstruct, real_fields)
        kl_loss = -0.5 *(1.0 + z_logvar - torch.square(z_mean) - torch.exp(z_logvar))
        kl_loss = torch.sum( kl_loss, axis=1 ) 
        kl_loss = torch.mean(kl_loss)
        
        loss = kl_loss + reconstruct_loss

        lossDict["vae_loss_kl"] = kl_loss
        lossDict["vae_loss_pix_me"] = nn.L1Loss()(reconstruct, real_fields)
        lossDict["vae_loss_pix_mse"] = nn.MSELoss()(reconstruct, real_fields)
        lossDict["loss"] = loss

        return lossDict

    def forward(self, input):
        z_mean, z_logvar, z = self.encoder(input)
        reconstruct = self.decoder(z)
        return reconstruct  
