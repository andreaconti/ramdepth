from pathlib import Path
import numpy as np
import sys
import re
from zipfile import ZipFile
import shutil
import os
import tempfile
import torch

__all__ = [
    "save_pfm",
    "read_pfm",
    "read_pfm_depth",
    "read_cam_file",
    "unzip",
]


def save_pfm(filename: str, image: np.ndarray, scale: float = 1) -> None:
    """Save a depth map from a .pfm file
    Args:
        filename: output .pfm file path string,
        image: depth map to save, of shape (H,W) or (H,W,C)
        scale: scale parameter to save
    """
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != "float32":
        raise Exception("Image dtype must be float32.")

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif (
        len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
    ):  # greyscale
        color = False
    else:
        raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    file.write("PF\n".encode("utf-8") if color else "Pf\n".encode("utf-8"))
    file.write("{} {}\n".format(image.shape[1], image.shape[0]).encode("utf-8"))

    endian = image.dtype.byteorder

    if endian == "<" or endian == "=" and sys.byteorder == "little":
        scale = -scale

    file.write(("%f\n" % scale).encode("utf-8"))

    image.tofile(file)
    file.close()


def read_pfm(filename: str) -> tuple[np.ndarray, float]:
    """Read a depth map from a .pfm file
    Args:
        filename: .pfm file path string
    Returns:
        data: array of shape (H, W, C) representing loaded depth map
        scale: float to recover actual depth map pixel values
    """
    file = open(filename, "rb")  # treat as binary and read-only
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf":  # depth is Pf
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width, 1)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def read_cam_file(path: str | Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Reads a file containing the camera intrinsics, extrinsics, max depth and
    min depth. valid for blended and dtu.

    Parameters
    ----------
    path: str
        path of the source file (something like ../00000000_cam.txt)

    Returns
    -------
    out: Tuple[np.ndarray, np.ndarray, float, float]
        respectively intrinsics, extrinsics, min depth and max depth
    """
    with open(path) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape(
        (4, 4)
    )
    intrinsics = np.fromstring(
        " ".join(lines[7:10]), dtype=np.float32, sep=" "
    ).reshape((3, 3))

    depth_range_info = [float(v) for v in lines[11].split()]
    if len(depth_range_info) == 4:
        depth_min = np.array(float(depth_range_info[0]), dtype=np.float32)
        depth_max = np.array(float(depth_range_info[-1]), dtype=np.float32)
    elif len(depth_range_info) == 3:
        depth_min = np.array(float(depth_range_info[0]), dtype=np.float32)
        depth_max = depth_min + depth_range_info[1] * depth_range_info[2]
    elif len(depth_range_info) == 2:
        depth_min = np.array(float(depth_range_info[0]), dtype=np.float32)
        depth_max = np.array(float(depth_range_info[1]), dtype=np.float32)
    else:
        raise RuntimeError(f"invalid depth range infor {depth_range_info} in {path}")

    return intrinsics, extrinsics, depth_min, depth_max


def read_pfm_depth(path: str | Path) -> np.ndarray:
    """
    Loads a blended or DTU depth map
    """
    depth = np.array(read_pfm(str(path))[0], dtype=np.float32)
    return depth


def unzip(path: str | Path, out: str | Path):
    with ZipFile(path) as file:
        file.extractall(out)


def download_unzip(url: str, out: str | Path, progress: bool = True):
    ftemp = os.path.join(tempfile.mkdtemp(prefix="download"), "temp_file")
    torch.hub.download_url_to_file(url, ftemp, progress=progress)
    unzip(ftemp, out)
    shutil.rmtree(os.path.dirname(ftemp))
