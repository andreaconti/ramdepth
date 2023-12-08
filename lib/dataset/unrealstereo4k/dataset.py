"""
Dataloader implementation for UnrealStereo, it works for both quarter
and full resolution
"""

from pathlib import Path
from typing import Literal, Callable
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from pipe import map, filter, chain
import numpy as np
from PIL import Image
import re
import pickle


class UnrealStereo4kDataset(Dataset):
    def __init__(
        self,
        root: Path | str = ".data/unrealstereo4k",
        split: Literal["test"] = "test",
        load_prevs: int | None = None,
        filter_scans: Callable[[str, str], bool] | None = None,
        stereo_as_prevs: bool = False,
        remove_sky: bool = True,
        scan_order: Literal[None, "pose", "pcd"] = "pcd",
        transform: Callable[[dict], dict] = lambda x: x,
    ):
        super().__init__()

        # fields
        self.root = Path(root)
        self.split = split
        self.load_prevs = load_prevs
        self.transform = transform
        self.stereo_as_prevs = stereo_as_prevs
        self.remove_sky = remove_sky
        self.scan_order = scan_order

        # load split
        with open(
            Path(__file__).parent.parent / f"_resources/unrealstereo_{split}.txt", "rt"
        ) as f:
            self.samples = list(
                f
                | map(lambda x: re.search("([0-9]+)/.*/([0-9]+)", x).groups())
                | map(lambda tup: (int(tup[0]), int(tup[1])))
            )

        # filter scans
        if filter_scans is not None:
            self.samples = list(
                self.samples | filter(lambda p: filter_scans(split, p[0]))
            )

        # order scans
        if self.scan_order is None:
            # default order
            self.order = {
                scan: [idx for s, idx in self.samples if s == scan]
                for scan in set(s[0] for s in self.samples)
            }
        else:
            # ordering relying on file order
            with open(
                Path(__file__).parent.parent
                / f"_resources/unrealstereo_order_{self.scan_order}.pkl",
                "rb",
            ) as order_file:
                self.order = pickle.load(order_file)
                new_samples = []
                for scan in set(s[0] for s in self.samples):
                    scan_samples = list(
                        self.samples
                        | filter(lambda s: s[0] == scan)
                        | map(lambda s: s[1])
                    )
                    new_samples.extend(
                        self.order[scan]
                        | filter(lambda s: s in scan_samples)
                        | map(lambda s: (scan, s))
                    )
                self.samples = new_samples

        # stereo as prevs
        self.order = {
            k: list(
                (
                    [(idx, "1"), (idx, "0")] if stereo_as_prevs else [(idx, "01")]
                    for idx in self.order[k]
                )
                | chain
            )
            for k in self.order
        }
        self.order = {k: list(zip(*v)) for k, v in self.order.items()}

        # remove samples without enough previous samples
        if load_prevs is not None:
            for scan, (ordering, _) in self.order.items():
                for to_rm in ordering[:load_prevs]:
                    try:
                        self.samples.remove((scan, to_rm))
                    except ValueError:
                        pass  # (scan, to_rm) not in the list

    def _load_depth(self, path, baseline):
        disparity = np.array(Image.open(path))[..., None] / 256.0
        depth = baseline / disparity.clip(min=2.0)
        if self.remove_sky:
            depth[disparity <= 1] = 0
        return self._pad_to_div_by(depth)

    def _load_rgb(self, path: Path):
        return self._pad_to_div_by(np.array(Image.open(path).convert("RGB")))

    def _load_extr(self, path: Path):
        with open(path, "rt") as f:
            intrins = np.fromstring(f.readline(), dtype=np.float32, sep=" ")
            intrins = intrins.reshape(3, 3)
            extrins = np.eye(4, 4, dtype=np.float32)
            extrins[:3, :4] = np.fromstring(
                f.readline(), dtype=np.float32, sep=" "
            ).reshape(3, 4)
        return extrins

    def _pad_to_div_by(self, x, *, div_by=8):
        # compute padding
        if isinstance(x, Image.Image):
            w, h = x.size
        elif isinstance(x, np.ndarray):
            h, w, _ = x.shape
        else:
            raise ValueError("Image or np.ndarray")

        new_h = int(np.ceil(h / div_by)) * div_by
        new_w = int(np.ceil(w / div_by)) * div_by
        pad_t, pad_b = self._split_pad(new_h - h)
        pad_l, pad_r = self._split_pad(new_w - w)

        # return PIL or np.ndarray
        if isinstance(x, Image.Image):
            return F.pad(x, (pad_l, pad_t, pad_r, pad_b), padding_mode="edge")
        elif isinstance(x, np.ndarray):
            return np.pad(x, [(pad_t, pad_b), (pad_l, pad_r), (0, 0)], mode="edge")

    def _split_pad(self, pad):
        if pad % 2 == 0:
            return pad // 2, pad // 2
        else:
            pad_1 = pad // 2
            pad_2 = (pad // 2) + 1
            return pad_1, pad_2

    def _synth_intrins(self, img):
        h, w, _ = img.shape
        return np.array(
            [[w / 2, 0.0, w / 2], [0.0, h / 2, h / 2], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _baseline(self, extr0, extr1):
        return np.abs(extr0[0, -1] - extr1[0, -1])

    def _load_single(self, seq, idx, view="0", end=""):
        root = self.root / f"{seq:0>5}"
        other_view = "0" if view == "1" else "1"
        out = {
            "image": (img := self._load_rgb(root / f"Image{view}/{idx:0>5}.png")),
            "intrinsics": (intr := self._synth_intrins(img)),
            "position": (
                pos := self._load_extr(root / f"Extrinsics{view}/{idx:0>5}.txt")
            ),
            "gt": self._load_depth(
                root / f"Disp{view}/{idx:0>5}.png",
                intr[0, 0]
                * self._baseline(
                    pos,
                    self._load_extr(root / f"Extrinsics{other_view}/{idx:0>5}.txt"),
                ),
            ),
        }
        return {k + end: v for k, v in out.items()}

    def _load_stereo(self, seq, idx, view, end=""):
        if view in ["0", "1"]:
            return self._load_single(seq, idx, view, end)
        else:
            return self._load_single(seq, idx, "0", end) | self._load_single(
                seq, idx, "1", "_other" + end
            )

    def _load_sample(self, seq, idx):
        samples_order, samples01 = self.order[seq]
        sample_idx = samples_order.index(idx)
        if not self.load_prevs:
            return self._load_stereo(
                seq,
                samples_order[sample_idx],
                samples01[sample_idx],
            )
        else:
            out = {}
            for i, prev_idx in enumerate(
                range(sample_idx + 1, sample_idx - self.load_prevs, -1)
            ):
                out |= self._load_stereo(
                    seq,
                    samples_order[prev_idx],
                    samples01[prev_idx],
                    end=f"_prev_{i - 1}" if i != 0 else "",
                )
            return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        seq, idx = self.samples[index]
        return self.transform(self._load_sample(seq, idx))
