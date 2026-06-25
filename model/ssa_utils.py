# model/ssa_utils.py

import torch

def sparse_topk_mask(mask, ratio=0.2):

    B,C,H,W = mask.shape

    flat = mask.view(B, -1)

    k = max(1, int(flat.shape[1] * ratio))

    topk_vals = torch.topk(flat, k, dim=1)[0]

    threshold = topk_vals[:, -1]

    sparse_mask = mask * (
        mask >= threshold.view(B,1,1,1)
    )

    return sparse_mask