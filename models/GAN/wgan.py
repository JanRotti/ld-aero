import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from modules.GAN.discriminator.discriminator import Discriminator
from modules.GAN.generator.generator import Generator

class WGAN(pl.LightningModule):
    def __init__(self, config, image_key="image", train_imbalance=0.5, gp_weight=10.0, label_noise=0.0):
        super().__init__()
        self.config = config
        config['discr_activation'] = nn.Identity()
        self.gp_weight = gp_weight
        self.label_noise = label_noise
        self.train_imbalance = train_imbalance
        self.image_key = image_key
        self.generator = Generator(**config)
        self.discriminator = Discriminator(**config)
        self.automatic_optimization = False

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        return x

    def _interpolate(self, real, fake):
        alpha = torch.rand(real.shape[0], 1, 1, 1).to(self.device)
        alpha = alpha.expand(real.shape)
        interpolated = alpha * real + (1 - alpha) * fake
        return interpolated

    def _wasserstein_distance(self, real, fake):
        return torch.mean(real * fake)

    def _gradient_penalty(self, real, fake):
        interpolated = self._interpolate(real, fake)
        interpolated = torch.autograd.Variable(interpolated, requires_grad = True)
        score = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=score,
            inputs=interpolated,
            grad_outputs=torch.ones(score.shape).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _discriminator_step(self, input):
        x = self.get_input(input, self.image_key)
        b, c, h, w = x.shape
        _, opt_d = self.optimizers()
        opt_d.zero_grad()

        latents = torch.randn(b, self.generator.latent_dim).to(self.device)
        fake = self.generator(latents)

        labelNoise = torch.abs(torch.randn(b) * self.label_noise)  
        valid = (torch.ones(b) - labelNoise).to(self.device)
        fake = -torch.ones(b).to(self.device)

        real_score = torch.squeeze(self.discriminator(x))
        fake_score = torch.squeeze(self.discriminator(fake))

        gp_loss = self._gradient_penalty(x, fake)

        real_score = self._wasserstein_distance(real_score, valid)
        fake_score = self._wasserstein_distance(fake_score, fake)
        loss = -real_score + fake_score + self.gp_weight * gp_loss

        self.manual_backward(loss)
        opt_d.step()

        loss_disc = {
            "disc/loss": loss,
            "disc/reals_score": real_score,
            "disc/fakes_score": fake_score,
            "disc/gp_loss": gp_loss,
            "disc/wasserstein" : real_score - fake_score
        }
        return loss_disc

    def _generator_step(self, input):
        self.discriminator.eval()
        x = self.get_input(input, self.image_key)
        b, c, h, w = x.shape
        opt_g, _ = self.optimizers()

        opt_g.zero_grad()

        latents = torch.randn(b, self.generator.latent_dim).to(self.device)
        fake = self.generator(latents)
        fake_score = torch.squeeze(self.discriminator(fake))
        
        loss = self._wasserstein_distance(fake_score, -torch.ones(b).to(self.device))

        self.manual_backward(loss)
        opt_g.step()
        self.discriminator.train()
        return {"gen/loss": loss}

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        b, c, h, w = x.shape
        losses = self._discriminator_step(batch)
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
        x = x.to(self.device)
        b, c, h, w = x.shape
        latent = torch.randn(b, self.generator.latent_dim).to(self.device)
        xrec = self.generator(latent)
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