import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from dataset import (
    BlendedMVSDataModule,
    TartanairDataModule,
    UnrealStereo4kDataModule,
)


@pytest.mark.parametrize(
    "dataset", ["blendedmvs", "tartanair", "unrealstereo4k"]
)
def test_load_data(dataset):
    dm = {
        "blendedmvs": BlendedMVSDataModule,
        "tartanair": TartanairDataModule,
        "unrealstereo4k": UnrealStereo4kDataModule,
    }[dataset]()
    dm.prepare_data()
    dm.setup("test")
    dl = dm.test_dataloader()
    ex = next(iter(dl))
