import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
参考文章: 
https://zhuanlan.zhihu.com/p/563661713 
"""



def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float):
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = torch.arange(timesteps + 1, dtype=torch.float32)
    alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999)



class GaussianDiffusion(nn.Module):
    """
    A self-contained scheduler that keeps all diffusion constants as buffers so
    they are saved / moved with the model.  API 兼容原先 `DDPMNoiseScheduler`.
    """
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device = torch.device("cpu"),
        t_sampler: str = "uniform",      # ["uniform", "lognormal"]
        t_log_mean: float = -0.5,        # lognormal  μ（
        t_log_sigma: float = 1.0,        # σ
        t_mix_uniform_p: float = 0.05,   
        t_clip_quantile: float = 0.999,  
    ):
        super().__init__()
        self.timesteps = timesteps
        self.device = torch.device(device) if device is not None else None
        self.t_sampler = t_sampler
        self.t_log_mean = float(t_log_mean)
        self.t_log_sigma = float(t_log_sigma)
        self.t_mix_uniform_p = float(t_mix_uniform_p)
        self.t_clip_quantile = float(t_clip_quantile)

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule {beta_schedule}")

        # precalculate constant, avoid redundant calculation 
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

        # calculations for q(x_t | x_{t-1}) 
        self.register_buffer("alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        with torch.no_grad():
            self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
            self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
            self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - self.alphas_cumprod))
            self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod))
            self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod - 1))

            # calculations for q(x_{t-1} | x_t, x_0)
            self.register_buffer("posterior_variance", betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

            # log calculation clipped
            self.register_buffer("posterior_log_variance_clipped", torch.log(self.posterior_variance.clamp(min=1e-20)))
            self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
            self.register_buffer("posterior_mean_coef2", (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

    # get the param of given timestep 
    def _extract(self, arr, t, x_shape):
        B = t.shape[0]
        out = arr.gather(0, t.to(arr.device)).view(B, *([1] * (len(x_shape) - 1)))
        return out

    def sample_timesteps(self, B: int):
        if self.t_sampler == "uniform":
            return torch.randint(0, self.timesteps, (B,), device=self.betas.device, dtype=torch.long)

        if self.t_sampler == "lognormal":
            dist = torch.distributions.LogNormal(
                torch.tensor(self.t_log_mean, device=self.betas.device),
                torch.tensor(self.t_log_sigma, device=self.betas.device)
            )
            y = dist.sample((B,))  
            q = dist.icdf(torch.tensor(self.t_clip_quantile, device=self.betas.device))
            s = (y / (q + 1e-12)).clamp(0.0, 1.0)  # s in [0,1]
            t_ln = (s * (self.timesteps - 1)).floor().long()

            if self.t_mix_uniform_p > 0:
                t_unif = torch.randint(0, self.timesteps, (B,), device=self.betas.device, dtype=torch.long)
                mask = (torch.rand(B, device=self.betas.device) < self.t_mix_uniform_p)
                t_ln = torch.where(mask, t_unif, t_ln)

            return t_ln

        raise ValueError(f"Unknown t_sampler {self.t_sampler}")

    # forward diffusion q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # get the mean and variance of q(x_t | x_0)
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        var = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_var = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, var, log_var

    # compute x_0 from x_t and pred noise 
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # get the mean and variance of q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = self._extract(self.posterior_variance, t, x_t.shape)
        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    # get the mean and variance of p(x_{t-1} | x_t)
    @torch.no_grad()
    def p_mean_variance(self, model, x_t, t, cond, clip_x0=True):
        pred_noise = model(x_t, t, cond)               
        x0_pred = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_x0:
            x0_pred = x0_pred.clamp(-1.0, 1.0)
        mean, var, log_var = self.q_posterior_mean_variance(x0_pred, x_t, t)
        return mean, var, log_var, pred_noise

    # denoise step 
    @torch.no_grad()
    def p_sample(self, model, x_t, t, cond, clip_x0=True):
        mean, _, log_var, _ = self.p_mean_variance(model, x_t, t, cond, clip_x0)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * (0.5 * log_var).exp() * noise


    # denoise
    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond, device=None, clip_x0=True):
        device = device or self.betas.device
        x_t = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, cond, clip_x0)
        return x_t

    # training loss 
    def train_losses(self, model, x_start, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred_noise = model(x_noisy, t)
        return F.mse_loss(pred_noise, noise)