import torch
from torchvision import transforms as T

__all__ = ["normalize_to_tensor"]

_img_process = T.Compose(
    [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
_depth_process = T.ToTensor()


def normalize_to_tensor(ex):
    for key, value in ex.items():
        if key.startswith("image"):
            ex[key] = _img_process(value)
        elif key.startswith("gt"):
            ex[key] = _depth_process(value)
    return ex


def prepare_input(ex):
    """
    Takes in input a batch from a dataloader and outputs the data processed
    for network inference
    """
    n_src = max(int(k.split("_")[-1]) for k in ex.keys() if "_prev" in k) + 1
    target = ex["image"]
    sources = torch.stack([ex[f"image_prev_{i}"] for i in range(n_src)], 2)
    poses = torch.stack(
        [
            ex["position"] @ torch.linalg.inv(ex[f"position_prev_{i}"])
            for i in range(n_src)
        ],
        1,
    )
    intrinsics = torch.stack(
        [ex["intrinsics"]] + [ex[f"intrinsics_prev_{i}"] for i in range(n_src)],
        1,
    )
    return {
        "target": target,
        "sources": sources,
        "poses": poses,
        "intrinsics": intrinsics,
    }
