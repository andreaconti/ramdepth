import torch
from torch import Tensor

__all__ = ["mae", "rmse", "perc_l1_upper_thresh" "compute_metrics"]


def compute_metrics(pred: Tensor, gt: Tensor):
    mask = gt > 0
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
