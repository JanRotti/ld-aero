import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from modules.GAN.discriminator.discriminator import ClassifyingDiscriminator
from modules.GAN.generator.generator import ConditionedGenerator

class ACGAN(pl.LightningModule):
    def __init__(self, config, image_key="image", label_key="label", train_imbalance=0.5, label_noise=0.0):
        super().__init__()
        self.config = config
        self.train_imbalance = train_imbalance
        self.image_key = image_key
        self.label_key = label_key
        self.label_noise = label_noise
        self.generator = ConditionedGenerator(**config)
        self.discriminator = ClassifyingDiscriminator(**config)
        self.automatic_optimization = False
        self.adversarial_loss = nn.BCELoss()
        tmp = nn.CosineSimilarity()
        self.auxiliary_loss = lambda x1,x2: tmp(x1, x2).mean()
        self.save_hyperparameters()

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        return x

    def _discriminator_step(self, input):
        x = self.get_input(input, self.image_key)
        label = self.get_input(input, self.label_key)
        b, c, h, w = x.shape
        _, opt_d = self.optimizers()
        opt_d.zero_grad()

        latents = torch.randn(b, self.generator.latent_dim).to(self.device)
        fake = self.generator(latents, label) # TODO: dont use label -> random generation of latent space

        real_score, real_label = self.discriminator(x)
        real_score = torch.squeeze(real_score)
        fake_score, fake_label = self.discriminator(fake)
        fake_score = torch.squeeze(fake_score)

        labelNoise = torch.abs(torch.randn(b) * self.label_noise)  
        valid = (torch.ones(b) - labelNoise).to(self.device)
        fake = torch.zeros(b).to(self.device)

        real_score = self.adversarial_loss(real_score, valid)
        fake_score = self.adversarial_loss(fake_score, fake)
        real_class_score = self.auxiliary_loss(real_label, label)
        fake_class_score = self.auxiliary_loss(fake_label, label)
        loss = (real_score + fake_score + real_class_score + fake_class_score) / 2
        wasserstein = real_score - fake_score

        self.manual_backward(loss)
        opt_d.step()

        loss_disc = {
            "disc/loss": loss,
            "disc/reals_score": real_score,
            "disc/reals_class_score": real_class_score,
            "disc/fakes_class_score": fake_class_score,
            "disc/fakes_score": fake_score,
            "wasserstein": wasserstein,
        }
        return loss_disc

    def _generator_step(self, input):
        x = self.get_input(input, self.image_key)
        label = self.get_input(input, self.label_key)
        b, c, h, w = x.shape
        opt_g, _ = self.optimizers()

        opt_g.zero_grad()

        latents = torch.randn(b, self.generator.latent_dim).to(self.device)
        fake = self.generator(latents, label) # TODO: dont use label -> random generation of latent space
        fake_score, fake_label = self.discriminator(fake)
        fake_score = torch.squeeze(fake_score)
        fake_class_score = self.auxiliary_loss(fake_label, label)
        gLoss = self.adversarial_loss(fake_score, torch.ones(b).to(self.device))

        self.manual_backward(gLoss)
        opt_g.step()
        return {"gen/loss": gLoss}

    def training_step(self, batch, batch_idx):
        losses = self._discriminator_step(batch)
        b, c, h, w = self.get_input(batch, self.image_key).shape
        if torch.rand(1) <= self.train_imbalance:
            gen_losses = self._generator_step(batch)
            losses.update(gen_losses)
        self.log_dict(losses, prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=b)
        return None

    def configure_optimizers(self):
        lr_gen = self.config.get('lr_gen', 0.0002)
        lr_dis = self.config.get('lr_dis', 0.0002)
        b1 = self.config.get('b1', 0.9)
        b2 = self.config.get('b2', 0.999)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_gen, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_dis, betas=(b1, b2))
        return opt_g, opt_d    

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        l = self.get_input(batch, self.label_key)
        x, l = x.to(self.device), l.to(self.device)
        b, c, h, w = x.shape
        latent = torch.randn(b, self.generator.latent_dim).to(self.device)
        xrec = self.generator(latent, l)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
            log["samples"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x