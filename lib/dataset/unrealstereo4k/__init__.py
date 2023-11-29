from .dataset import UnrealStereo4kDataset
from pytorch_lightning import LightningDataModule
from pathlib import Path
from typing import Callable, Literal
from multiprocessing import cpu_count
from torch.utils.data import DataLoader

__all__ = ["UnrealStereo4kDataModule"]


class UnrealStereo4kDataModule(LightningDataModule):
    def __init__(
        self,
        # dataset specific
        root: Path | str = ".data/unrealstereo4k",
        load_prevs: int | None = 0,
        filter_scans: Callable[[str, str], bool] | None = None,
        stereo_as_prevs: bool = True,
        scan_order: Literal[None, "pose", "pcd"] = "pcd",
        remove_sky: bool = False,
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
        self.stereo_as_prevs = stereo_as_prevs
        self.scan_order = scan_order
        self.remove_sky = remove_sky

    def prepare_data(self) -> None:
        # TODO: prepare data download
        pass

    def setup(self, stage: str | None = None):
        self._test_ds = UnrealStereo4kDataset(
            self.root,
            split="test",
            load_prevs=self.load_prevs,
            stereo_as_prevs=self.stereo_as_prevs,
            filter_scans=self.filter_scans,
            remove_sky=self.remove_sky,
            scan_order=self.scan_order,
            transform=self.eval_transform if self.eval_transform else lambda x: x,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )
