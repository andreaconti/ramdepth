"""
Utility functions to load the DTU dataset
"""

from pathlib import Path
from typing import Callable, Literal

import numpy as np
from PIL import Image
from torch import Tensor
from plyfile import PlyData
from functools import partial
import torchvision.transforms.functional as F
import scipy.io

from torch.utils.data import DataLoader
import torchdata.datapipes as dp
from multiprocessing import cpu_count
from pytorch_lightning import LightningDataModule
from ..utils import read_pfm_depth, read_cam_file

__all__ = [
    "load_dtu",
    "DTUDataModule",
    "dtu_scans",
]


class DTUDataModule(LightningDataModule):
    def __init__(
        self,
        # dataset specific
        root: Path | str = ".data/dtu",
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

    def test_dataloader(self):
        return load_dtu(
            self.root,
            "test",
            load_prevs=self.load_prevs,
            filter_scans=self.filter_scans,
            batch_size=self.batch_size,
            transform=self.eval_transform,
            num_workers=self.num_workers,
        )


def dtu_scans(split: str):
    return {"train": train_scans, "val": val_scans, "test": test_scans}[split]()


def load_dtu(
    root: str | Path,
    split: Literal["test"] = "test",
    *,
    load_prevs: int = 0,
    filter_scans: Callable[[str, str], bool] | None = None,
    batch_size: int = 1,
    transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
    num_workers: int = cpu_count() // 2,
    used_scans=dtu_scans,
):
    # retrieve sequence
    seqs = used_scans(split)
    if filter_scans:
        seqs = [s for s in seqs if filter_scans(split, s)]

    # build available samples
    metas = build_list(root, seqs)
    examples = []
    for meta in metas:
        ref_view = meta["ref_view"]
        n_src = len(meta["src_views"])
        curr_frame = datapath_files(root, meta["scan"], ref_view, meta["light_idx"])
        if load_prevs >= 1 and load_prevs <= n_src:
            prev_frames = [
                datapath_files(root, meta["scan"], s, meta["light_idx"])
                for s in meta["src_views"]
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
                examples.append(
                    datapath_files(root, meta["scan"], src_view, meta["light_idx"])
                )

    ds = dp.map.SequenceWrapper(examples)
    ds = ds.map(partial(_decode, split=split))
    ds = ds.map(transform) if transform else ds
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


# utilities


def _read_single(ex, label="", idx=None):
    # load data
    intrins, extrins, depth_min, depth_max = read_cam_file(
        ex[f"cam_info{label}"] if idx is None else ex[f"cam_info{label}"][idx]
    )
    gt = read_pfm_depth(
        ex[f"depth{label}"] if idx is None else ex[f"depth{label}"][idx]
    )
    gt[(gt < depth_min) & (gt > depth_max)] = 0.0
    image = Image.open(ex[f"image{label}"] if idx is None else ex[f"image{label}"][idx])
    image = np.array(image)

    # resize
    hi, wi = image.shape[:2]
    hg, wg = gt.shape[:2]
    if hi != hg and wi != wg:
        gt = F.center_crop(F.to_tensor(gt[::2, ::2]), (512, 640)).numpy()[0, ..., None]
        intrins[:2] *= 0.5
        intrins[0, 2] *= 640 / 800
        intrins[1, 2] *= 512 / 600

    out = {
        "image": image,
        "intrinsics": intrins,
        "position": extrins,
        "gt": gt,
        "min_depth": depth_min,
        "max_depth": depth_max,
    }

    return out


def _load_pcd(ex):
    out = {"scan_id": ex["scan_id"], "id": ex["id"]}
    if ex["light_id"] is not None:
        out["light_id"] = ex["light_id"]
    if ex["pcd"] is not None:
        pcd = PlyData.read(ex["pcd"])
        out["pcd"] = np.stack(
            [pcd["vertex"]["x"], pcd["vertex"]["y"], pcd["vertex"]["z"]], -1
        )
    if ex["obs_mask"] is not None:
        obs_data = scipy.io.loadmat(ex["obs_mask"])
        out["pcd_obs_mask"] = obs_data["ObsMask"]
        out["pcd_bounding_box"] = obs_data["BB"]
        out["pcd_resolution"] = obs_data["Res"]
    if ex["ground_plane"] is not None:
        out["pcd_ground_plane"] = scipy.io.loadmat(ex["ground_plane"])["P"]
    return out


def _decode(ex: dict, split: str = "train"):
    out = _read_single(ex, "")
    if "cam_info_prev" in ex:
        if not isinstance(ex["cam_info_prev"], (list, tuple)):
            out |= {k + f"_prev": v for k, v in _read_single(ex, f"_prev").items()}
        else:
            for i in range(len(ex["cam_info_prev"])):
                out |= {
                    k + f"_prev_{i}": v
                    for k, v in _read_single(ex, f"_prev", i).items()
                }
    out |= _load_pcd(ex)
    return out


def datapath_files(root: Path | str, scan: str, idx: int, light_idx: int = None):
    """
    Takes in input the root of the DTU dataset and returns a dictionary containing
    the paths to the files of a single scan, with the specified view_id and light_id
    (this last one is used only when ``split`` isn't test)

    Parameters
    ----------
    datapath: path
        path to the root of the dtu dataset
    scan: str
        name of the used scan, say scan1
    view_id: int
        the index of the specific image in the scan
    split: train, val or test
        which split of DTU must be used to search the scan for
    light_id: int
        the index of the specific lightning condition index

    Returns
    -------
    out: Dict[str, Path]
        returns a dictionary containing the paths taken into account
    """

    root = Path(root)
    if light_idx is not None:
        root = root / "train_data"
        return {
            "image": root
            / f"Rectified/{scan}_train/rect_{idx + 1:0>3}_{light_idx}_r5000.png",
            "cam_info": root / f"Cameras_1/{idx:0>8}_cam.txt",
            "depth_mask": root / f"Depths_raw/{scan}/depth_visual_{idx:0>4}.png",
            "depth": root / f"Depths_raw/{scan}/depth_map_{idx:0>4}.pfm",
            "pcd": None,
            "obs_mask": None,
            "ground_plane": None,
            "scan_id": scan,
            "light_id": light_idx,
            "id": idx,
        }
    else:
        scan_idx = int(scan[4:])
        root = root / "test_data"
        return {
            "image": root / f"{scan}/images/{idx:0>8}.jpg",
            "cam_info": root / f"{scan}/cams_1/{idx:0>8}_cam.txt",
            "depth_mask": root.parent
            / f"train_data/Depths_raw/{scan}/depth_visual_{idx:0>4}.png",
            "depth": root.parent
            / f"train_data/Depths_raw/{scan}/depth_map_{idx:0>4}.pfm",
            "pcd": root.parent
            / f"SampleSet/MVS Data/Points/stl/stl{scan_idx:0>3}_total.ply",
            "obs_mask": root.parent
            / f"SampleSet/MVS Data/ObsMask/ObsMask{scan_idx}_10.mat",
            "ground_plane": root.parent
            / f"SampleSet/MVS Data/ObsMask/Plane{scan_idx}.mat",
            "scan_id": scan,
            "light_id": None,
            "id": idx,
        }


def build_list(root: str | Path, scans: list):
    metas = []

    root = Path(root)
    for scan in scans:
        # find pair file
        pair_file_test = root / f"test_data/{scan}/pair.txt"
        pair_file_train = root / "train_data/Cameras_1/pair.txt"
        if not pair_file_test.exists():
            if not pair_file_train.exists():
                raise ValueError(f"scan {scan} not found")
            pair_file = pair_file_train
            split = "train"
        else:
            pair_file = pair_file_test
            split = "test"

        # use pair file
        with open(pair_file, "rt") as f:
            num_viewpoint = int(f.readline())
            for _ in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                if split == "train":
                    for light_idx in range(7):
                        metas.append(
                            {
                                "scan": scan,
                                "light_idx": light_idx,
                                "ref_view": ref_view,
                                "src_views": src_views,
                            }
                        )
                else:
                    metas.append(
                        {
                            "scan": scan,
                            "light_idx": None,
                            "ref_view": ref_view,
                            "src_views": src_views,
                        }
                    )

    return metas


# SPLITS


def train_scans():
    return [
        "scan2",
        "scan3",  #
        "scan5",  #
        "scan6",
        "scan7",
        "scan8",
        "scan14",
        "scan16",
        "scan17",  #
        "scan18",
        "scan19",
        "scan20",
        "scan21",  #
        "scan22",
        "scan28",  #
        "scan30",
        "scan31",
        "scan36",
        "scan37",  #
        "scan38",  #
        "scan39",
        "scan40",  #
        "scan41",
        "scan42",
        "scan43",  #
        "scan44",
        "scan45",
        "scan46",
        "scan47",
        "scan50",
        "scan51",
        "scan52",
        "scan53",
        "scan55",
        "scan56",  #
        "scan57",
        "scan58",
        "scan59",  #
        "scan60",
        "scan61",
        "scan63",
        "scan64",
        "scan65",
        "scan66",  #
        "scan67",  #
        "scan68",
        "scan69",
        "scan70",
        "scan71",
        "scan72",
        "scan74",
        "scan76",
        "scan82",  #
        "scan83",
        "scan84",
        "scan85",
        "scan86",  #
        "scan87",
        "scan88",
        "scan89",
        "scan90",
        "scan91",
        "scan92",
        "scan93",
        "scan94",
        "scan95",
        "scan96",
        "scan97",
        "scan98",
        "scan99",
        "scan100",
        "scan101",
        "scan102",
        "scan103",
        "scan104",
        "scan105",
        "scan106",  #
        "scan107",
        "scan108",
        "scan109",
        "scan111",
        "scan112",
        "scan113",
        "scan115",
        "scan116",
        "scan117",  #
        "scan119",
        "scan120",
        "scan121",
        "scan122",
        "scan123",
        "scan124",
        "scan125",
        "scan126",
        "scan127",
        "scan128",
    ]


def val_scans():
    return [
        "scan1",
        "scan4",
        "scan9",
        "scan10",
        "scan11",
        "scan12",
        "scan13",
        "scan15",
        "scan23",
        "scan24",
        "scan29",
        "scan32",
        "scan33",
        "scan34",
        "scan48",
        "scan49",
        "scan62",
        "scan75",
        "scan77",
        "scan110",
        "scan114",
        "scan118",
    ]


def test_scans():
    return [
        "scan1",
        "scan4",
        "scan9",
        "scan10",
        "scan11",
        "scan12",
        "scan13",
        "scan15",
        "scan23",
        "scan24",
        "scan29",
        "scan32",
        "scan33",
        "scan34",
        "scan48",
        "scan49",
        "scan62",
        "scan75",
        "scan77",
        "scan110",
        "scan114",
        "scan118",
    ]
