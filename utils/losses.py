# utils/losses.py

import torch.nn.functional as F

def reconstruction_loss(pred, target):
    return F.l1_loss(pred, target)

def kl_penalty(*latents):
    """Compute KL loss as L2 norm over multiple latent vectors."""
    return sum([z.pow(2).mean() for z in latents]) / len(latents)

def lpips_loss_fn(lpips_model, preds, targets):
    """Compute average LPIPS loss over paired predictions and targets."""
    return sum([lpips_model(p, t).mean() for p, t in zip(preds, targets)]) / len(preds)