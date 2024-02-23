import torch
import torch.nn as nn
import lightning as pl
import torch.nn.functional as F

from modules.decoder import Decoder
from modules.encoder import Encoder
from modules.distribution import DiagonalGaussianDistribution

from .base import Autoencoder

class VAE(Autoencoder):
    def __init__(self,
                 config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=0.00001,
                 learning_rate=0.001,
                 betas=(0.9,0.99),
                 **kwargs,
                 ):
        
        super(**kwargs).__init__()
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.betas = betas
        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)
        
        assert config["double_z"]
        self.embed_dim = config["z_channels"]
        self.quant_conv = torch.nn.Conv2d(2*config["z_channels"], 2*self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, config["z_channels"], 1)
        self.kl_weight = kl_weight
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        aeloss = nn.MSELoss()(reconstructions, inputs)
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0] * self.kl_weight
        loss = aeloss + kl_loss

        log_dict_ae = {"train/total_loss": loss.clone().detach().mean(),
                       "train/kl_loss": kl_loss.detach().mean(),
                       "train/rec_loss": aeloss.detach().mean(),
                       }
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss = nn.MSELoss()(reconstructions, inputs)
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0] * self.kl_weight
        loss = aeloss + kl_loss

        log_dict_ae = {"val/total_loss": loss.clone().detach().mean(),
                       "val/kl_loss": kl_loss.detach().mean(),
                       "val/rec_loss": aeloss.detach().mean(),
                       }
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=self.betas)
        return opt_ae

    def get_last_layer(self):
        return self.decoder.conv_out.weight
