import torch
from torch import Tensor

__all__ = [
    "mae",
    "rmse",
    "perc_l1_upper_thresh",
    "compute_metrics",
    "depth_to_disp",
]


def depth_to_disp(depth, intrins1, pose1, pose2, min_depth=0.1):
    mask = depth > 0
    b = torch.abs(pose1[..., 0, -1] - pose2[..., 0, -1])
    f = intrins1[..., 0, 0]
    disp = torch.zeros_like(depth)
    disp[mask] = f * b / depth[mask].clip(min=min_depth)
    return disp


def compute_metrics(pred: Tensor, gt: Tensor, mask: Tensor | None):
    if mask is not None:
        pred, gt = pred[mask], gt[mask]

    return {
        "mae": mae(pred, gt),
        "rmse": rmse(pred, gt),
    } | {
        f"perc_l1_upper_thresh_{i}": perc_l1_upper_thresh(pred, gt, i)
        for i in [1, 2, 3, 4, 8]
    }


def mae(pred: Tensor, gt: Tensor) -> Tensor:
    return torch.mean(torch.abs(pred - gt))


def rmse(pred: Tensor, gt: Tensor) -> Tensor:
    return torch.sqrt(torch.mean(torch.square(pred - gt)))


def perc_l1_upper_thresh(pred: Tensor, gt: Tensor, thresh: float):
    return torch.mean((torch.abs(pred - gt) > thresh).float())
