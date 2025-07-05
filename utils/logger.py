# utils/logger.py

import wandb
from torchvision.utils import make_grid
import torch

def init_wandb(config):
    wandb.init(project="font-disentangle-week2", config=config)

def log_images(imgA, recA, crossBA, gt_crossBA, imgB, recB, crossAB, gt_crossAB):
    grid = make_grid(
        torch.cat([
            imgA[:4], recA[:4], crossBA[:4], gt_crossBA[:4],
            imgB[:4], recB[:4], crossAB[:4], gt_crossAB[:4]
        ], dim=0),
        nrow=4, normalize=True, scale_each=True
    )
    wandb.log({"recon_vis": [wandb.Image(grid, caption="A(row1): GT→Recon→CrossBA→GT_BA | B(row2): GT→Recon→CrossAB→GT_AB")]})

def log_single_image(img, recon):
    """Log single input and reconstruction to wandb."""
    grid = make_grid(
        torch.cat([img[:4], recon[:4]], dim=0),
        nrow=4, normalize=True, scale_each=True
    )
    wandb.log({"reconstruction_vis": [wandb.Image(grid, caption="Top: Input | Bottom: Recon")]})

def log_loss(step, loss):
    wandb.log({"loss": loss}, step=step)

def log_epoch(epoch, avg_loss):
    wandb.log({"epoch_loss": avg_loss, "epoch": epoch})

def log_losses(step, loss_dict):
    wandb.log(loss_dict, step=step)

def log_latents(step, **kwargs):
    wandb.log({k: wandb.Histogram(v.detach().cpu()) for k, v in kwargs.items()}, step=step)