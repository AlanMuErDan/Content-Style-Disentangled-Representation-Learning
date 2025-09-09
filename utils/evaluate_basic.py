import torch 
import torch.nn.functional as F

def compute_psnr(img1, img2):
    """Compute PSNR between two images, img1 and img2 are (B, 1, H, W) or (B, C, H, W)"""
    mse = F.mse_loss(img1, img2, reduction='none').mean(dim=[1, 2, 3])  # (B,)
    psnr = -10 * torch.log10(mse + 1e-8)
    return psnr.mean().item()


def compute_ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    """Simplified SSIM for grayscale image batches. img1, img2: (B, 1, H, W)"""
    mu1 = F.avg_pool2d(img1, 3, 1, 0)
    mu2 = F.avg_pool2d(img2, 3, 1, 0)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 0) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 0) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 0) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()

def compute_metrics(pred, target):
    # pred, target: (B, C, H, W) and normalized to [0, 1]
    return {
        "L1": F.l1_loss(pred, target).item(),
        "L2": F.mse_loss(pred, target).item(),
        "SSIM": compute_ssim(pred, target),
        "PSNR": compute_psnr(pred, target),
    }