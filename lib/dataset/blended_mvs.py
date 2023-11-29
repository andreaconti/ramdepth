"""
To load the BlendedMVS dataset
"""

from typing import Callable, Literal
from torch import Tensor
from multiprocessing import cpu_count
import torchdata.datapipes as dp
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
from pathlib import Path
from pytorch_lightning import LightningDataModule
from .utils import read_pfm_depth, read_cam_file

# api

__all__ = ["BlendedMVSDataModule"]


class BlendedMVSDataModule(LightningDataModule):
    def __init__(
        self,
        # dataset specific
        root: Path | str = ".data/blended-mvs",
        filter_scans: Callable[[str, str], bool] | None = None,
        load_prevs: int = 0,
        # common
        batch_size: int = 1,
        eval_transform: Callable[[dict], dict] | None = None,
        num_workers=cpu_count() // 2,
    ):
        super().__init__()
        self.root = root
        self.filter_scans = filter_scans
        self.load_prevs = load_prevs
        self.batch_size = batch_size
        self.eval_transform = eval_transform
        self.num_workers = num_workers

    def prepare_data(self):
        # TODO: prepare automatic download of data
        pass

    def test_dataloader(self):
        return load_blended_mvs(
            self.root,
            "test",
            load_prevs=self.load_prevs,
            filter_scans=self.filter_scans,
            batch_size=self.batch_size,
            transform=self.eval_transform,
            num_workers=self.num_workers,
        )


def _used_scans():
    return {"test": test_scans}


def load_blended_mvs(
    root: str | Path,
    split: Literal["test"] = "test",
    *,
    load_prevs: int = 0,
    filter_scans: Callable[[str, str], bool] | None = None,
    batch_size: int = 1,
    transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
    num_workers: int = cpu_count() // 2,
    used_scans=_used_scans,
):
    # retrieve sequence
    seqs = used_scans()[split]()
    if filter_scans:
        seqs = [s for s in seqs if filter_scans(split, s)]

    # build available samples
    metas = build_list(root, seqs)
    examples = []
    for meta in metas:
        ref_view = meta["ref_view"]
        n_src = len(meta["src_views"])
        curr_frame = datapath_files(root, meta["scan"], ref_view)
        if load_prevs >= 1 and load_prevs <= n_src:
            prev_frames = [
                datapath_files(root, meta["scan"], s) for s in meta["src_views"]
            ]
            n = (n_src // load_prevs) * load_prevs
            for chunk in np.array_split(prev_frames[:n], n_src // load_prevs):
                chunk = {k: [dic[k] for dic in chunk] for k in chunk[0]}
                examples.append(curr_frame | {n + "_prev": v for n, v in chunk.items()})
                if split != "train":
                    break
        elif load_prevs == 0:
            examples.append(curr_frame)
            for src_view in meta["src_views"]:
                examples.append(datapath_files(root, meta["scan"], src_view))

    ds = dp.map.SequenceWrapper(examples)
    ds = ds.map(_decode)
    ds = ds.map(transform) if transform else ds
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


# utilities


def _decode(ex: dict):
    # decode curr
    intrins, extrins, depth_min, depth_max = read_cam_file(ex["cam_info"])
    gt = read_pfm_depth(ex["depth"])
    gt[(gt < depth_min) & (gt > depth_max)] = 0.0
    out = {
        "image": np.array(Image.open(ex["image"])),
        "intrinsics": intrins,
        "position": extrins,
        "gt": gt,
        "min_depth": depth_min,
        "max_depth": depth_max,
    }

    # decode prev
    if "cam_info_prev" in ex:
        if not isinstance(ex["cam_info_prev"], (list, tuple)):
            intrins, extrins, depth_min, depth_max = read_cam_file(ex["cam_info_prev"])
            gt = read_pfm_depth(ex["depth_prev"])
            gt[(gt < depth_min) & (gt > depth_max)] = 0.0
            out |= {
                "image_prev": np.array(Image.open(ex["image_prev"])),
                "intrinsics_prev": intrins,
                "position_prev": extrins,
                "gt_prev": gt,
                "min_depth_prev": depth_min,
                "max_depth_prev": depth_max,
            }
        else:
            for i in range(len(ex["cam_info_prev"])):
                intrins, extrins, depth_min, depth_max = read_cam_file(
                    ex["cam_info_prev"][i]
                )
                gt = read_pfm_depth(ex["depth_prev"][i])
                gt[(gt < depth_min) & (gt > depth_max)] = 0.0
                out |= {
                    f"image_prev_{i}": np.array(Image.open(ex["image_prev"][i])),
                    f"intrinsics_prev_{i}": intrins,
                    f"position_prev_{i}": extrins,
                    f"gt_prev_{i}": gt,
                    f"min_depth_prev_{i}": depth_min,
                    f"max_depth_prev_{i}": depth_max,
                }

    return out


def build_list(datapath: str, scans: list):
    metas = []
    for scan in scans:
        pair_file = "cams/pair.txt"

        with open(os.path.join(datapath, scan, pair_file)) as f:
            num_viewpoint = int(f.readline())
            for _ in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                metas.append(
                    {"scan": scan, "ref_view": ref_view, "src_views": src_views}
                )
    return metas


def datapath_files(datapath: Path | str, scan: str, idx: int) -> dict:
    """
    Takes in input the root of the Blended MVS dataset and returns a dictionary containing
    the paths to the files of a single scan element

    Parameters
    ----------
    datapath: path
        path to the root of the dtu dataset
    scan: str
        name of the used scan, say scan1
    idx: int
        the index of the specific image in the scan
    Returns
    -------
    out: Dict[str, Path]
        returns a dictionary containing the paths taken into account
    """

    root = Path(datapath)
    return {
        "image": root / f"{scan}/blended_images/{idx:0>8}.jpg",
        "cam_info": root / f"{scan}/cams/{idx:0>8}_cam.txt",
        "depth": root / f"{scan}/rendered_depth_maps/{idx:0>8}.pfm",
    }


def test_scans() -> list[str]:
    return [
        "5b7a3890fc8fcf6781e2593a",
        "5c189f2326173c3a09ed7ef3",
        "5b950c71608de421b1e7318f",
        "5a6400933d809f1d8200af15",
        "59d2657f82ca7774b1ec081d",
        "5ba19a8a360c7c30c1c169df",
        "59817e4a1bd4b175e7038d19",
    ]
