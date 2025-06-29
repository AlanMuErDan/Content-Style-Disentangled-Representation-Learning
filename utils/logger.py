# utils/logger.py
import wandb
from torchvision.utils import make_grid
import torch

def init_wandb(config):
    wandb.init(project="font-disentangle", config=config)

def log_images(imgA, recA, crossBA, gt_crossBA, imgB, recB, crossAB, gt_crossAB):
    grid = make_grid(
        torch.cat([
            imgA[:4], recA[:4], crossBA[:4], gt_crossBA[:4],
            imgB[:4], recB[:4], crossAB[:4], gt_crossAB[:4]
        ], dim=0),
        nrow=4, normalize=True, scale_each=True
    )
    wandb.log({"recon_vis": [wandb.Image(grid, caption="A(row1): GT→Recon→CrossBA→GT_BA | B(row2): GT→Recon→CrossAB→GT_AB")]})

def log_loss(step, loss):
    wandb.log({"loss": loss}, step=step)

def log_epoch(epoch, avg_loss):
    wandb.log({"epoch_loss": avg_loss, "epoch": epoch})