import torch
import torch.nn as nn
import lightning as pl
import torch.nn.functional as F

from modules.decoder import Decoder
from modules.encoder import Encoder

from .base import Autoencoder

class VanillaAE(Autoencoder):
    """
    Vanilla Convolutional Autoencoder
    """
    def __init__(self,
                 config,
                 ignore_keys=[],
                 image_key=None,
                 monitor=None,
                 learning_rate=0.0001,
                 betas=(0.9, 0.99),
                 ):
        
        super().__init__()

        self.image_key = image_key
        self.betas = betas
        self.learning_rate = learning_rate
        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        z = self.encode(input)
        dec = self.decode(z)
        return dec

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions = self(inputs)
        
        aeloss = nn.MSELoss()(reconstructions, inputs)
        loss = aeloss

        log_dict_ae = {
                        "train/loss": loss.clone().detach(),
                        "train/rec_loss": aeloss.detach(),
                       }
       
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions = self(inputs)
        aeloss = F.mse_loss(reconstructions, inputs)
        loss = aeloss

        log_dict_ae = {
                        "val/loss": loss.clone().detach(),
                        "val/rec_loss": aeloss.detach(),
                       }
                       
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters()),
                                  lr=self.learning_rate, betas=self.betas)
        return opt_ae

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            z = self.encode(x)
            xrec = self.decode(z)
            samples = self.decode(torch.randn_like(z))
            if x.shape[1] > 3:
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
                samples = self.to_rgb(samples)
            log["samples"] = samples
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log
    