from typing import Callable, Literal
from pathlib import Path
from torch import Tensor
import numpy as np
from torchdata import datapipes as dp
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from . import _webdataset_ext
from pytorch_lightning import LightningDataModule
from .utils._download import github_download_unzip_assets, root_data
import imageio.v3 as imageio
import json
import re


class TartanairDataModule(LightningDataModule):
    def __init__(
        self,
        root: str | Path = root_data("tartanair"),
        # dataset specific
        load_prevs: int = 0,
        filter_trajectories: list[str] | None = None,
        motion_ref_frame: Literal["cam", "ned"] = "cam",
        max_depth: float | None = 100.0,
        # dataloader specific
        batch_size: int = 1,
        eval_transform: Callable[[dict], dict] | None = None,
        num_workers: int = cpu_count() // 2,
    ):
        super().__init__()

        if load_prevs > 20:
            raise ValueError("load prevs > 20 not supported")

        self.root = root
        self.batch_size = batch_size
        self.eval_transform = eval_transform
        self.num_workers = num_workers
        self.filter_trajectories = filter_trajectories
        self.load_prevs = load_prevs
        self.motion_ref_frame = motion_ref_frame
        self.max_depth = max_depth

        if self.filter_trajectories is not None:
            with open(
                Path(__file__).parent / f"_resources/tartanair_test.json", "rt"
            ) as f:
                all_trajs = json.load(f)
            if not all(s in all_trajs for s in self.filter_trajectories):
                raise ValueError("some invalid trajectory names provided")

    def _filter_traj(self, split) -> str | list[str]:
        if self.filter_trajectories is None:
            return split
        else:
            with open(
                Path(__file__).parent / f"_resources/tartanair_{split}.json",
                "rt",
            ) as f:
                used_trajs = json.load(f)
            return [traj for traj in self.filter_trajectories if traj in used_trajs]

    def prepare_data(self):
        root = Path(self.root)
        root.mkdir(exist_ok=True, parents=True)
        files = [f.name for f in root.iterdir()]

        with open(Path(__file__).parent / f"_resources/tartanair_test.json", "rt") as f:
            test_scans = [s + ".tar" for s in json.load(f)]

        for scan in test_scans:
            if scan not in files:
                github_download_unzip_assets(
                    "andreaconti",
                    "ramdepth",
                    ["139715957", "139715955", "139715954"],
                    root,
                )
                break

    def setup(self, stage: Literal["test"] | None = None):
        if stage not in ["test", None]:
            raise ValueError(f"stage {stage} invalid")
        self._test_dl = load_tartanair(
            self.root,
            self._filter_traj("test"),
            load_prevs=self.load_prevs,
            motion_ref_frame=self.motion_ref_frame,
            max_depth=self.max_depth,
            batch_size=self.batch_size,
            transform=self.eval_transform,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return self._test_dl


def load_tartanair(
    root: str | Path,
    split: Literal["test"] | str | list[str] = "test",
    # dataset specific
    load_prevs: int = 0,
    motion_ref_frame: Literal["cam", "ned"] = "cam",
    max_depth: float | None = None,
    # dataloader specific
    batch_size: int = 1,
    transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
    num_workers: int = cpu_count() // 4,
) -> DataLoader:
    # split handling
    root = Path(root)
    split_src = str(Path(__file__).parent / "_resources/tartanair_test.json")

    if split == "test":
        sequences = [s + ".tar" for s in json.load(open(split_src.format(split), "rt"))]
    elif split in json.load(split_src, "rt"):
        sequences = [split]
    elif isinstance(split, (tuple, list)):
        if all(s in json.load(open(split_src, "rt")) for s in split):
            split = [s + ".tar" for s in split]
            sequences = split
        else:
            raise ValueError(f"seq in sequences {split} not available")

    ds = dp.iter.FileOpener([str(root / seq) for seq in sequences], mode="b")
    ds = (
        ds.load_from_tar_with_archive()
        .webdataset_with_archive()
        .load_prevs(prevs=load_prevs)
    )
    ds = ds.sharding_filter()
    ds = ds.map(_Decode(motion_ref_frame=motion_ref_frame, max_depth=max_depth))
    ds = ds.map(transform) if transform else ds
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


# Decoding Functions

_ned_to_cam = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


class _Decode:
    def __init__(
        self,
        motion_ref_frame: Literal["cam", "ned"] = "cam",
        max_depth: float | None = None,
    ):
        self.names = {
            "depth": "gt",
            "image": "image",
            "intrinsics": "intrinsics",
            "position": "position",
        }
        self.motion_ref_frame = motion_ref_frame
        self.max_depth = max_depth
        self._name_re = re.compile(
            r"\.(" + "|".join(self.names.keys()) + r")(_prev_\d+)?\.(json|jpg|png|npy)"
        )

    def rename(self, name):
        if match := self._name_re.match(name):
            name, prev, _ = match.groups()
            return self.names[name], self.names[name] + (
                prev if prev is not None else ""
            )
        else:
            return None

    def __call__(self, sample):
        out = {}
        for key, value in sample.items():
            match self.rename(key):
                case "gt", new_name:
                    depth = imageio.imread(value)[..., None].astype(np.float32)
                    drange = key.replace("depth", "depth_range").replace("png", "json")
                    sample[drange].seek(0)
                    depth_range = json.load(sample[drange])
                    vmin, vmax = depth_range["vmin"], depth_range["vmax"]
                    out[new_name] = vmin + (depth / 65535) * (vmax - vmin)
                    if self.max_depth is not None:
                        out[new_name] = np.where(
                            out[new_name] <= self.max_depth,
                            out[new_name],
                            0.0,
                        )
                case "image", new_name:
                    image = imageio.imread(value)
                    out[new_name] = image.copy()
                case "intrinsics", new_name:
                    value.seek(0)
                    out[new_name] = np.load(value).astype(np.float32)
                case "position", new_name:
                    value.seek(0)
                    pose = np.load(value).astype(np.float32)
                    if self.motion_ref_frame == "cam":
                        pose = _ned_to_cam.dot(pose).dot(_ned_to_cam.T)
                    out[new_name] = np.linalg.inv(pose)
        return out
