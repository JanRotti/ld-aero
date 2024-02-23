import lightning.pytorch as pl
import torch
import numpy as np

from modules.deepflow.encoder import Encoder
from modules.deepflow.encoder import Decoder
from modules.embedding.vector_quantizer import VectorQuantizer2 as VectorQuantizer

class DeepFlowVQVAE(pl.LightningModule):

    def __init__(self, config, latent_dim):
        super().__init__()
        
        self.encoder = Encoder(*config)
        self.decoder = Decoder(*config)
        self.q_layer = VectorQuantizer(64, latent_dim)
        self.latent_dim = latent_dim

    def configure_optimizers(self):
        self.learning_rate = 0.002
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.q_layer.parameters()),
                                  lr=lr, betas=(0.9, 0.99))
        return opt_ae

    def training_step(self, data):
        mse = nn.MSELoss()
        real_fields = data

        lossDict = {}
        
        z_mean, z_logvar, z = self.encoder([real_fields, None])
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
        real_fields = data

        lossDict = {}

        z_mean,z_logvar,z = self.encoder([real_fields,None])
        reconstruct = self.decoder(z)
        
        reconstruct_loss = mse(reconstruct, real_fields)
        kl_loss = -0.5 *(1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
        kl_loss = tf.reduce_sum( kl_loss, axis=1 ) 
        kl_loss = tf.reduce_mean(kl_loss)
        
        loss = kl_loss + reconstruct_loss

        lossDict["vae_loss_kl"] = kl_loss
        lossDict["vae_loss_pix_me"] = tf.keras.losses.MeanAbsoluteError()(reconstruct,real_fields)
        lossDict["vae_loss_pix_mse"] = tf.keras.losses.MeanSquaredError()(reconstruct,real_fields)
        lossDict["loss"] = loss

        return lossDict2
