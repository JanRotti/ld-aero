import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from functools import partial

from modules.util import extract, generate_cosine_schedule, generate_linear_schedule, default

class EMA():
    def __init__(self, decay):
        self.decay = decay
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)


class DDPM(pl.LightningModule):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, in_channels, *image_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        image_size (tuple): image size tuple (H, W)
        in_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        metric (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """
    def __init__(self,
        model,
        image_size,
        in_channels,
        betas=None,
        timesteps=1000,
        use_ema=True,
        ema_decay=0.9999,
        ema_start=5000,
        ema_update_rate=1,
        learning_rate=0.001,
        opt_betas=(0.9, 0.99),
        parameterization="eps",
        metric="l2",
        image_key="image",
        beta_schedule="linear",
        ignore_keys=[],
        ckpt_path=None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ckpt_path", "image_key", "ignore_keys", "model"])
        
        self.betas = betas
        self.timesteps = timesteps
        self.image_size = image_size
        self.in_channels = in_channels
        self.model = model
        self.beta_schedule = beta_schedule

        # Check parameters
        assert(parameterization in ["eps", "x0"], "'parameterization' must be 'eps' or 'x0'.")
        if self.betas is not None:
           self.timesteps = len(betas)
        self.register_schedule()

        # Optimizer parameters
        self.learning_rate = learning_rate
        self.opt_betas = opt_betas

        # Utility
        self.image_key = image_key
        self.parameterization = parameterization

        # Exponential Moving Average
        self.use_ema = use_ema
        self.ema_model = copy.deepcopy(model)
        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.metric = nn.L1Loss() if metric == "l1" else nn.MSELoss()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)


    def register_schedule(self):
        if self.betas is None:
            if self.beta_schedule == "linear":
                self.betas = generate_linear_schedule(self.timesteps, 1e-4, 2e-2)
            elif self.beta_schedule == "cosine":
                self.betas = generate_cosine_schedule(self.timesteps, 0.008)
            else:
                raise ValueError("Unknown beta schedule")
        
        betas = self.betas
        alphas = 1.0 -betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))


    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)


    @torch.no_grad()
    def remove_noise(self, x, t, y=None, use_ema=True):
        if use_ema:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )

    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.in_channels, *self.image_size, device=device)
        
        for t in range(self.timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            
            x = self.remove_noise(x, t_batch, y, use_ema)
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        return x

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.in_channels, *self.image_size, device=device)
        diffusion_sequence = [x.cpu().detach()]
        
        for t in range(self.timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            
            diffusion_sequence.append(x.cpu().detach())
        
        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )   

    def loss(self, x, t, noise=None):
        
        noise = default(noise, lambda: torch.randn_like(x))
        log_prefix = 'train' if self.training else 'val'
        
        perturbed_x = self.perturb_x(x, t, noise)
        estimation = self.model(perturbed_x, t)
        
        target = noise if self.parameterization == "eps" else x
        diff = torch.abs(target - estimation)

        est_loss = self.metric(estimation, target)

        loss = est_loss 

        loss_dict = {}
        loss_dict.update({f"{log_prefix}/loss": loss.clone().detach()})

        return loss, loss_dict

    def forward(self, x, noise=None):
        b, c, h, w = x.shape
        device = x.device

        if h != self.image_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.image_size[0]:
            raise ValueError("image width does not match diffusion parameters")
        
        t = torch.randint(0, self.timesteps, (b,), device=device)
        return self.loss(x, t, noise=noise)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        b, c, h, w = x.shape
        loss, loss_dict = self(x)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=b)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        b, c, h, w = x.shape
        loss, loss_dict = self(x)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=b)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.update_ema()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=self.opt_betas)
        return opt

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        return x

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        b, c, h, w = x.shape
        x = x.to(self.device)
        if not only_inputs:
            samples = self.sample(b, self.device)
            if x.shape[1] > 3:
                x = self.to_rgb(x)
                samples = self.to_rgb(samples)
            log["samples"] = samples
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x - x.min())/(x.max() - x.min()) - 1. 
        return x