import torch
import torch.nn.functional as F

def _maybe_norm(x, do_norm: bool):
    return F.normalize(x, dim=1) if do_norm else x

def info_nce_pairwise(anchor, positive, negatives, tau: float, do_norm: bool) -> torch.Tensor:
    """
    对每个样本 i：
      anchor[i] 的正样本是 positive[i]（一一对应）
      负样本池为 negatives（形状 [M, D]，一般 M = 2B）
    返回一个标量 loss（batch 均值）
    """
    anchor = _maybe_norm(anchor, do_norm)
    positive = _maybe_norm(positive, do_norm)
    negatives = _maybe_norm(negatives, do_norm)

    # 正样本 logit：逐样本点积（B, 1）
    pos_logit = torch.sum(anchor * positive, dim=1, keepdim=True) / tau  # (B,1)

    # 负样本 logit：与整个负样本池做矩阵乘（B, M）
    neg_logits = anchor @ negatives.t() / tau

    # 拼接 [pos | negs]，正样本标签恒为 0
    logits = torch.cat([pos_logit, neg_logits], dim=1)  # (B, 1+M)
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)
