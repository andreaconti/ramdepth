import matplotlib.pyplot as plt
import torch

__all__ = ["plot_sample"]


def plot_sample(
    ex: torch.Tensor,
    pred: torch.Tensor,
    mask: bool = True,
):
    n_src = max(int(k.split("_")[-1]) for k in ex.keys() if "_prev" in k) + 1

    # sources
    for idx in range(n_src):
        plt.subplot(1, n_src + 3, idx + 1)
        plt.axis("off")
        plt.title(f"Source #{idx}")
        img = ex[f"image_prev_{idx}"][0].permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img)

    # reference
    plt.subplot(1, n_src + 3, idx + 2)
    plt.axis("off")
    plt.title("Reference")
    img = ex[f"image"][0].permute(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)

    # prediction
    gt_mask = ex["gt"] > 0
    vmin, vmax = ex["gt"][gt_mask].min(), ex["gt"].max()
    if mask:
        pred = torch.where(gt_mask, pred, vmax)
    plt.subplot(1, n_src + 3, idx + 3)
    plt.axis("off")
    plt.title("Prediction")
    plt.imshow(pred[0, 0], cmap="magma_r", vmin=vmin, vmax=vmax)

    # ground truth
    plt.subplot(1, n_src + 3, idx + 4)
    plt.axis("off")
    plt.title("Ground Truth")
    gt = ex["gt"]
    if mask:
        gt = torch.where(gt_mask, gt, vmax)
    plt.imshow(gt[0, 0], cmap="magma_r", vmin=vmin, vmax=vmax)
