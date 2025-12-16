import torch
import torch.nn.functional as F

def _maybe_norm(x, do_norm: bool):
    return F.normalize(x, dim=1) if do_norm else x

def info_nce_pairwise(anchor, positive, negatives, tau: float, do_norm: bool) -> torch.Tensor:

    anchor = _maybe_norm(anchor, do_norm)
    positive = _maybe_norm(positive, do_norm)
    negatives = _maybe_norm(negatives, do_norm)


    pos_logit = torch.sum(anchor * positive, dim=1, keepdim=True) / tau  # (B,1)

    neg_logits = anchor @ negatives.t() / tau

    logits = torch.cat([pos_logit, neg_logits], dim=1)  # (B, 1+M)
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)
