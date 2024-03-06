import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from functools import partial
from einops import reduce

from modules.util import extract, default, linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule

def identity(t, *args, **kwargs):
    return t

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

class DDPM(pl.LightningModule):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, in_channels, *image_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        image_size (tuple): image size tuple (H, W)
        in_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        metric (string): loss type, "l1" or "l2"
    """
    def __init__(self,
        model: nn.Module,
        image_size,
        in_channels,
        betas=None,
        timesteps=1000,
        learning_rate=0.001,
        opt_betas=(0.9, 0.99),
        objective="eps",
        metric="l2",
        image_key="image",
        beta_schedule="linear",
        ignore_keys=[],
        ckpt_path=None,
        auto_normalize=True,
        sampling_timesteps=None,
        ddim_sampling_eta=0.,
        offset_noise_strength = 0., # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        **kwargs
        ):

        super().__init__()
        self.save_hyperparameters(ignore=["ckpt_path", "image_key", "ignore_keys", "model", *ignore_keys])
        
        self.betas = betas
        self.timesteps = timesteps
        self.image_size = image_size
        self.in_channels = in_channels
        self.model = model
        self.beta_schedule = beta_schedule
        self.sampling_timesteps = sampling_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.offset_noise_strength = offset_noise_strength
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma
        self.self_condition = False
        self.objective = objective

        # Check parameters
        assert objective in ["eps", "x0", "v"], "'objective' must be 'eps' or 'x0' or 'v'."
        if self.betas is not None:
           self.timesteps = len(betas)
        if self.sampling_timesteps is None:
            self.sampling_timesteps = self.timesteps   
        assert self.sampling_timesteps <= self.timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.timesteps
        

        self.register_schedule()

        # Optimizer parameters
        self.learning_rate = learning_rate
        self.opt_betas = opt_betas

        # Utility
        self.image_key = image_key

        self.step = 0

        self.metric = nn.L1Loss() if metric == "l1" else nn.MSELoss()
        
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

    def register_schedule(self):
        if self.betas is None:
            if self.beta_schedule == "linear":
                self.betas = linear_beta_schedule(self.timesteps)
            elif self.beta_schedule == "cosine":
                self.betas = cosine_beta_schedule(self.timesteps, 0.008)
            elif self.beta_schedule == "sigmoid":
                self.betas = sigmoid_beta_schedule(self.timesteps, 0.1, 0.9)
            else:
                raise ValueError("Unknown beta schedule")

        betas = self.betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer("reciprocal_sqrt_alphas", torch.sqrt(1 / alphas))
        self.register_buffer("reciprocal_minus_one_sqrt_alphas", torch.sqrt(1 / alphas))
        self.register_buffer("remove_noise_coeff", betas / torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("sigma", torch.sqrt(betas))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        snr = alphas_cumprod / (1 - alphas_cumprod)
    
        maybe_clipped_snr = snr.clone()
        if self.min_snr_loss_weight:
            maybe_clipped_snr = torch.clamp(maybe_clipped_snr, max=self.min_snr_gamma)
        if self.objective == "eps":
            self.register_buffer("loss_weight", maybe_clipped_snr / snr)
        elif self.objective == "x0":
            self.register_buffer("loss_weight", maybe_clipped_snr)
        elif self.objective == "v":
            self.register_buffer("loss_weight", maybe_clipped_snr / (snr + 1))

    def predict_ref_from_noise(self, x_t, t, noise=None):
        return (
            extract(self.reciprocal_sqrt_alphas, t, x_t.shape) * x_t -
            extract(self.reciprocal_minus_one_sqrt_alphas, t, x_t.shape) * noise
        )
    
    def predict_noise_from_ref(self, x_t, t, x0):
       return (
            (extract(self.reciprocal_sqrt_alphas, t, x_t.shape) * x_t - x0) / \
            extract(self.reciprocal_minus_one_sqrt_alphas, t, x_t.shape)
        ) 

    def predict_v(self, x0, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * x0
        )

    def predict_ref_from_v(self, x_t, t, v):
       return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        ) 

    def q_posterior(self, x0, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict(self, x, t, y=None, clip_x0=False):
        model_out = self.model(x, t, y)
        clip = partial(torch.clamp, min = -1., max = 1.) if clip_x0 else identity
        
        if self.objective == "eps":
            pred_noise = model_out
            x0 = self.predict_ref_from_noise(x, t, pred_noise)
            x0 = clip(x0)
        elif self.objective == "x0":
            x0 = model_out
            x0 = clip(x0)
            pred_noise = self.predict_noise_from_ref(x, t, x0)
        elif self.objective == "v":
            v = model_out
            x0 = self.predict_ref_from_v(x, t, v)
            x0 = clip(x0)
            pred_noise = self.predict_ref_from_nois(x, t, x0)
        
        return pred_noise, x0

    def p_mean_variance(self, x, t, y=None, clip_denoised=True):
        pred_noise, x0 = self.predict(x, t, y)

        if clip_denoised:
            x0.clamp(min=-1., max=1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x0=x0, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x0

    @torch.inference_mode()
    def p_sample(self, x, t: int, y=None):
        b, *_ = x.shape
        batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)
        model_mean, _, model_log_variance, x0 = self.p_mean_variance(x=x, t=batched_times, y=y, clip_denoised=True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x0

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        batch, device = shape[0], self.device
        img = torch.randn(shape, device=device)
        imgs = [img]
        x0 = None
        for t in reversed(range(0, self.timesteps)):
            y = x0 if self.self_condition else None
            img, x0 = self.p_sample(img, t, y)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps-1, steps=sampling_timesteps+1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x0 = None
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.int)
            y = x0 if self.self_condition else None
            pred_noise, x0, *_ = self.predict(img, time_cond, y, clip_x0=True)

            if time_next < 0:
                img = x0
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x0 * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        image_size, in_channels = self.image_size, self.in_channels
        return self.p_sample_loop((batch_size, in_channels, image_size[0], image_size[1]), return_all_timesteps=return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.timesteps-1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x0 = None

        for i in reversed(range(0, t)):
            y = x0 if self.self_condition else None
            img, x0 = self.p_sample(img, i, y)

        return img

    def q_sample(self, x0, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x0))
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def p_losses(self, x0, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))
        
        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)
        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x0.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample
        x = self.q_sample(x0=x0, t=t, noise=noise)
        # if doing self-conditioning, 50% of the time, predict x0 from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        y = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                pred_noise, y = self.predict(x, t) # self conditioning
                y.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, y)

        if self.objective == 'eps':
            target = noise
        elif self.objective == 'x0':
            target = x0
        elif self.objective == 'v':
            v = self.predict_v(x0, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w = img.shape
        device = img.device
        img_size = self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.timesteps, (b,), device=device)
        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        b, c, h, w = x.shape
        loss = self(x)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=b)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        b, c, h, w = x.shape
        loss = self(x)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=b)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=self.opt_betas)
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
            samples = self.sample(b)
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