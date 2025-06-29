import torch.nn.functional as F

def reconstruction_loss(pred, target):
    return F.l1_loss(pred, target)