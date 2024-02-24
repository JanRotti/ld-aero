import torch
import torch.nn as nn
import lightning as pl
import torch.nn.functional as F

from modules.decoder import Decoder
from modules.encoder import Encoder
from modules.embedding import VectorQuantizer2 as VectorQuantizer
from modules.distribution import DiagonalGaussianDistribution

from .base import Autoencoder

class VQVAE(Autoencoder):
    def __init__(self,
                 config,
                 num_embeddings=100,
                 commitment_scale=0.25,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key=None,
                 colorize_nlabels=None,
                 monitor=None,
                 learning_rate=0.001,
                 betas=(0.9,0.99),
                 final_activation=None,
                 ):
        
        super().__init__()

        self.image_key = image_key
        self.commitment_scale = commitment_scale
        self.learning_rate = learning_rate
        self.betas = betas

        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)

        self.embed_dim = config["z_channels"]
        self.quant_conv = torch.nn.Conv2d(config["z_channels"], self.embed_dim, 1)
        self.quantize = VectorQuantizer(num_embeddings, self.embed_dim, self.commitment_scale)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, config["z_channels"], 1)
        self.final_activation = final_activation if final_activation else nn.Identity()

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x, return_loss=False):
        h = self.encoder(x)
        z = self.quant_conv(h)
        quant, emb_loss, _ = self.quantize(z)
        if return_loss:
            return quant, emb_loss
        return quant

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, return_loss=False):
        if return_loss:
            z, emb_loss = self.encode(input, return_loss)
            dec = self.decode(z)
            return dec, emb_loss
        else:
            z = self.encode(input)
            dec = self.decode(z)
            return dec

    def get_input(self, batch, k):
        #x = batch[k]
        x, y = batch
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, emb_loss = self(inputs, return_loss=True)

        aeloss = nn.MSELoss()(reconstructions, inputs)
        vq_loss = emb_loss
        loss = aeloss + vq_loss

        log_dict_ae = {"loss": loss.clone().detach().mean(),
                       "vq_loss": vq_loss.detach().mean(),
                       "rec_loss": aeloss.detach().mean(),
                       }
        #self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, emb_loss = self(inputs, return_loss=True)
        aeloss = nn.MSELoss()(reconstructions, inputs)
        vq_loss = emb_loss
        loss = aeloss + vq_loss

        log_dict_ae = {"loss": loss.clone().detach().mean(),
                       "vq_loss": vq_loss.detach().mean(),
                       "rec_loss": aeloss.detach().mean(),
                       }
        #self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        betas = self.betas
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=betas)
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
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(z))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log