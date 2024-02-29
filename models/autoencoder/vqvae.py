import torch
import torch.nn as nn
import lightning as pl
import torch.nn.functional as F

from modules.decoder import Decoder, PSDecoder
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
                 kl_weight=0.00001,
                 learning_rate=0.001,
                 betas=(0.9,0.99),
                 **kwargs,
                 ):
        
        super().__init__()
        self.save_hyperparameters(ignore=["ckpt_path", "image_key","ignore_keys"])
        self.image_key = image_key
        self.commitment_scale = commitment_scale
        self.learning_rate = learning_rate
        self.betas = betas
        self.kl_weight = kl_weight

        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)

        self.embed_dim = config["z_channels"]
        assert config["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*config["z_channels"], 2*self.embed_dim, 1)
        self.quantize = VectorQuantizer(num_embeddings, self.embed_dim, self.commitment_scale)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, config["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x, return_loss=False, sample_posterior=True):
        h = self.encoder(x)
        z = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(z)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        quant, emb_loss, _ = self.quantize(z)
        if return_loss:
            return quant, emb_loss, posterior.kl()
        return quant

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, return_loss=False):
        if return_loss:
            z, emb_loss, kl_loss = self.encode(input, return_loss)
            dec = self.decode(z)
            return dec, emb_loss, kl_loss
        else:
            z = self.encode(input)
            dec = self.decode(z)
            return dec

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, emb_loss, kl_loss = self(inputs, return_loss=True)
        
        kl_loss = torch.mean(kl_loss) * self.kl_weight
        aeloss = nn.MSELoss()(reconstructions, inputs)
        vq_loss = emb_loss

        loss = aeloss + vq_loss + kl_loss

        log_dict_ae = {
                       "train/loss": loss.clone().detach(),
                       "train/vq_loss": vq_loss.detach(),
                       "train/kl_loss": kl_loss.detach(),
                       "train/rec_loss": aeloss.detach(),
                       }

        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, emb_loss, kl_loss = self(inputs, return_loss=True)
        kl_loss = kl_loss * self.kl_weight
        aeloss = nn.MSELoss()(reconstructions, inputs)
        vq_loss = emb_loss
        loss = aeloss + vq_loss + kl_loss

        log_dict_ae = {
                       "val/loss": loss.clone().detach(),
                       "val/kl_loss": kl_loss.detach(),
                       "val/vq_loss": vq_loss.detach(),
                       "val/rec_loss": aeloss.detach(),
                       }

        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=False)
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
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
                samples = self.to_rgb(samples)
            log["samples"] = samples 
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log


class VQVAE_Attention(Autoencoder):
    def __init__(self,
                 config,
                 num_embeddings=100,
                 commitment_scale=0.25,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key=None,
                 kl_weight=0.00001,
                 learning_rate=0.001,
                 betas=(0.9, 0.99),
                 **kwargs,
                 ):
        
        super().__init__(config=config,num_embeddings=num_embeddings,commitment_scale=commitment_scale,ckpt_path=ckpt_path,ignore_keys=ignore_keys,image_key=image_key,kl_weight=kl_weight,learning_rate=learning_rate,betas=betas,**kwargs)
        self.decoder = AttentionDecoder(**config)