# models/flow_matching.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class ProbabilityPath(nn.Module):
    def sample_time(self, B: int, device) -> torch.Tensor:
        raise NotImplementedError

    def forward_path(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def target_velocity(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss_weight(self, t: torch.Tensor, x_shape=None) -> torch.Tensor:
        return torch.ones_like(t, dtype=torch.float32, device=t.device)


class LinearRFPath(ProbabilityPath):
    def __init__(self, t_sampler: str = "uniform",
                 ln_mu: float = -0.5, ln_sigma: float = 1.0,
                 mix_unif_p: float = 0.05, clip_q: float = 0.999):
        super().__init__()
        self.t_sampler = str(t_sampler)
        self.ln_mu = float(ln_mu)
        self.ln_sigma = float(ln_sigma)
        self.mix_unif_p = float(mix_unif_p)
        self.clip_q = float(clip_q)

    @torch.no_grad()
    def sample_time(self, B: int, device) -> torch.Tensor:
        if self.t_sampler == "uniform":
            return torch.rand(B, device=device)

        if self.t_sampler == "lognormal":
            dist = torch.distributions.LogNormal(
                torch.tensor(self.ln_mu, device=device),
                torch.tensor(self.ln_sigma, device=device)
            )
            y = dist.sample((B,))
            q = dist.icdf(torch.tensor(self.clip_q, device=device))
            s = (y / (q + 1e-12)).clamp(0.0, 1.0)
            t_ln = s
            if self.mix_unif_p > 0:
                t_unif = torch.rand(B, device=device)
                mask = (torch.rand(B, device=device) < self.mix_unif_p)
                t_ln = torch.where(mask, t_unif, t_ln)
            return t_ln

        raise ValueError(f"Unknown t_sampler: {self.t_sampler}")

    def forward_path(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        return (1.0 - t) * x0 + t * eps

    def target_velocity(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        return eps - x0

    def loss_weight(self, t: torch.Tensor, x_shape=None) -> torch.Tensor:
        return torch.ones_like(t, dtype=torch.float32, device=t.device)



class FlowMatching(nn.Module):
    def __init__(
        self,
        path: ProbabilityPath = None,
        t_epsilon: float = 1e-5,
        ode_solver: str = "heun",   # ["euler", "heun"]
        ode_steps: int = 50,
    ):
        super().__init__()
        self.path = path if path is not None else LinearRFPath()
        self.t_epsilon = float(t_epsilon)
        self.ode_solver = str(ode_solver).lower()
        self.ode_steps = int(ode_steps)

    @torch.no_grad()
    def sample_times(self, B: int, device) -> torch.Tensor:
        return self.path.sample_time(B, device)
    
    @torch.no_grad()
    def sample_timesteps(self, B: int, device=None):
        if device is None:
            device = next(self.parameters()).device if len(list(self.parameters())) else torch.device("cpu")
        return self.sample_times(B, device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor = None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x0)
        return self.path.forward_path(x0, eps, t)

    def train_losses(self, model, x0: torch.Tensor, cond=None, eps: torch.Tensor = None,
                     times: torch.Tensor = None, reduction: str = "mean"):
        B = x0.size(0)
        device = x0.device

        if times is None:
            t = self.sample_times(B, device)
        else:
            t = times

        if eps is None:
            eps = torch.randn_like(x0)

        x_t = self.path.forward_path(x0, eps, t)
        v_target = self.path.target_velocity(x0, eps, t, x_t)

        v_pred = model(x_t, t, cond) if cond is not None else model(x_t, t)

        w = self.path.loss_weight(t, x_shape=x0.shape)
        while w.dim() < v_pred.dim():
            w = w.unsqueeze(-1)

        loss_per = (v_pred - v_target).pow(2)
        loss_per = loss_per * w
        if reduction == "mean":
            loss = loss_per.mean()
        elif reduction == "sum":
            loss = loss_per.sum()
        else:
            loss = loss_per
        return loss



    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond=None, device=None,
                      t_start: float = 1.0, t_end: float = 0.0, clip_x0: bool = False):
        if device is None:
            device = next(model.parameters()).device
        B = shape[0]
        x = torch.randn(shape, device=device)  

        steps = max(1, self.ode_steps)
        ts = torch.linspace(t_start - self.t_epsilon, t_end + self.t_epsilon, steps + 1, device=device)

        for i in range(steps, 0, -1):
            t_hi = ts[i]
            t_lo = ts[i - 1]
            dt = (t_lo - t_hi)  

            t_in = torch.full((B,), t_hi.item(), device=device, dtype=torch.float32)
            if self.ode_solver == "euler":
                v = model(x, t_in, cond) if cond is not None else model(x, t_in)
                x = x + dt * v
            elif self.ode_solver == "heun":
                # predictor
                v1 = model(x, t_in, cond) if cond is not None else model(x, t_in)
                x_pred = x + dt * v1
                # corrector
                t_mid = torch.full((B,), (t_hi + t_lo).item()/2.0, device=device, dtype=torch.float32)
                v2 = model(x_pred, t_mid, cond) if cond is not None else model(x_pred, t_mid)
                x = x + dt * 0.5 * (v1 + v2)
            else:
                raise ValueError(f"Unknown ODE solver: {self.ode_solver}")

        if clip_x0:
            x = x.clamp(-1.0, 1.0)
        return x

    @torch.no_grad()
    def sample(self, model, shape, cond=None, device=None, **kwargs):
        return self.p_sample_loop(model, shape, cond, device, **kwargs)



class FM_CFG_Wrapper:
    def __init__(self, denoiser, cfg_scale: float):
        self.denoiser = denoiser
        self.cfg_scale = float(cfg_scale)

    @torch.no_grad()
    def __call__(self, x, t, c):
        c_uncond = torch.zeros_like(c)
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        c_in = torch.cat([c, c_uncond], dim=0)
        v = self.denoiser(x_in, t_in, c_in)
        B = x.size(0)
        v_cond, v_uncond = v[:B], v[B:]
        return v_uncond + self.cfg_scale * (v_cond - v_uncond)
