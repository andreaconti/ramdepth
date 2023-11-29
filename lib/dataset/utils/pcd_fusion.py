import numpy as np
import argparse
import torch
from plyfile import PlyData, PlyElement
from PIL import Image
import re
import torch.nn.functional as F
import itertools
from tqdm import tqdm
from pathlib import Path


def pcd_fusion(
    scan_folder: str,
    pred_folder: str,
    outpath: str,
    glb: float = 0.25,
    device: str | torch.device = "cpu",
    verbose: bool = True,
):

    image_files = find_all_images(scan_folder)
    pred_files = find_all_predictions(pred_folder)
    cams_files = find_all_cams(scan_folder)
    pair_file = find_pair_file(scan_folder)

    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    n_view = len(pair_data)

    # for each reference view and the corresponding source views
    ct2 = -1

    # store all views
    all_images = {}
    all_depths = {}
    all_intrinsics = {}
    all_extrinsics = {}
    for i in tqdm(image_files.keys(), desc="Loading Files", disable=not verbose):
        ref_img = read_img(image_files[i])
        ref_depth_est = read_pfm(pred_files[i])
        ref_intrinsics, ref_extrinsics = read_camera_parameters(cams_files[i])
        all_images[i] = ref_img
        all_depths[i] = ref_depth_est
        all_extrinsics[i] = ref_extrinsics
        all_intrinsics[i] = ref_intrinsics

    # pad views
    h, w = 0, 0
    for img in all_images.values():
        h, w = max(h, img.shape[0]), max(w, img.shape[1])
    for i in image_files.keys():
        depth = center_crop(all_depths[i], *all_images[i].shape[:2])
        all_intrinsics[i] = adjust_intrinsics(depth, all_intrinsics[i], h, w)
        all_depths[i] = pad(depth[..., None], h, w)[..., 0]
        all_images[i] = pad(all_images[i], h, w, mode="edge")
        if all_depths[i].shape[:2] != (h, w):
            print("!", all_depths[i].shape)

    # move all on device
    all_images = {k: torch.from_numpy(img) for k, img in all_images.items()}
    all_depths = {k: torch.from_numpy(d) for k, d in all_depths.items()}
    all_intrinsics = {k: torch.from_numpy(i).float() for k, i in all_intrinsics.items()}
    all_extrinsics = {k: torch.from_numpy(i).float() for k, i in all_extrinsics.items()}

    thre_left = -2
    thre_right = 2

    tot_iter = 10
    for iter in tqdm(range(tot_iter), desc="Composing pcd", disable=not verbose):
        thre = (thre_left + thre_right) / 2

        depth_est = torch.zeros((n_view, h, w)).to(device)
        geo_mask_all = []
        ref_id = 0

        for ref_view, src_views in pair_data:
            ct2 += 1

            ref_img = all_images[ref_view].to(device)
            ref_depth_est = all_depths[ref_view].to(device)
            ref_extrinsics = all_extrinsics[ref_view].to(device)
            ref_intrinsics = all_intrinsics[ref_view].to(device)

            n = 1 + len(src_views)
            src_intrinsics = torch.stack([all_intrinsics[v] for v in src_views]).to(
                device
            )
            src_extrinsics = torch.stack([all_extrinsics[v] for v in src_views]).to(
                device
            )
            src_depth_est = torch.stack([all_depths[v] for v in src_views]).to(device)
            n_src = len(src_views)
            ref_depth_est = ref_depth_est.unsqueeze(0).repeat(n_src, 1, 1)
            ref_intrinsics = ref_intrinsics.unsqueeze(0).repeat(n_src, 1, 1)
            ref_extrinsics = ref_extrinsics.unsqueeze(0).repeat(n_src, 1, 1)
            if n_src == 0:
                ref_id += 1
                continue
            masks, geo_mask, depth_reprojected, *_ = check_geometric_consistency(
                ref_depth_est,
                ref_intrinsics,
                ref_extrinsics,
                src_depth_est,
                src_intrinsics,
                src_extrinsics,
                10**thre * 4,
                10**thre * 1300,
            )

            geo_mask_sums = []
            for i in range(2, n):
                geo_mask_sums.append(masks[i - 2].sum(dim=0).int())
            geo_mask_sum = geo_mask.sum(dim=0)
            geo_mask = geo_mask_sum >= n

            for i in range(2, n):
                geo_mask = torch.logical_or(geo_mask, geo_mask_sums[i - 2] >= i)
            depth_est[ref_id] = (depth_reprojected.sum(dim=0) + ref_depth_est[0]) / (
                geo_mask_sum + 1
            )
            del masks
            geo_mask_all.append(geo_mask.float().mean().item())

            if iter == tot_iter - 1:
                ref_intrinsics = ref_intrinsics[0]
                ref_extrinsics = ref_extrinsics[0]
                depth_est_averaged = depth_est[ref_id].cpu().numpy()
                geo_mask = geo_mask.cpu().numpy()
                valid_points = geo_mask

                ref_img = ref_img.cpu().numpy()

                height, width = depth_est_averaged.shape[:2]
                x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

                x, y, depth = (
                    x[valid_points],
                    y[valid_points],
                    depth_est_averaged[valid_points],
                )
                color = ref_img[:, :, :][valid_points]
                xyz_ref = np.matmul(
                    np.linalg.inv(ref_intrinsics.cpu().numpy()),
                    np.vstack((x, y, np.ones_like(x))) * depth,
                )
                xyz_world = np.matmul(
                    np.linalg.inv(ref_extrinsics.cpu().numpy()),
                    np.vstack((xyz_ref, np.ones_like(x))),
                )[:3]
                vertexs.append(xyz_world.transpose((1, 0)))
                vertex_colors.append((color * 255).astype(np.uint8))
            ref_id += 1
        if np.mean(geo_mask_all) >= glb:
            thre_left = thre
        else:
            thre_right = thre

    vertexs = np.array(
        [tuple(v) for vt in vertexs for v in vt],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    vertex_colors = np.array(
        [tuple(v) for vt in vertex_colors for v in vt],
        dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, "vertex")
    PlyData([el]).write(outpath)


# UTILS


def read_pfm(file):
    file = open(file, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b"PF":
        color = True
    elif header == b"Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(rb"^(\d+)\s(\d+)\s$", file.readline())
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
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def bilinear_sampler(img, coords, mask=False):
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def find_pair_file(path) -> str:
    return next(Path(path).glob("**/pair.txt"))


def find_all_images(path) -> dict[int, str]:
    images = itertools.chain(Path(path).glob("**/*.jpg"), Path(path).glob("**/*.png"))
    images = list(map(str, images))
    out_images = {}
    try:
        idx2prefix = next(Path(path).glob("**/index2prefix.txt"))
        with open(idx2prefix, "rt") as f:
            f.readline()
            for idx, rel_path in map(lambda x: x.split(), f):
                out_images[int(idx)] = next(filter(lambda p: rel_path in p, images))
    except StopIteration:
        out_images = {int(Path(p).stem): p for p in images}

    return out_images


def find_all_cams(path) -> dict[int, str]:
    cams = {}
    for cam in map(str, Path(path).glob("**/*_cam.txt")):
        idx = int(re.match(".*/([0-9]+)_cam\.txt.*", cam).groups()[0])
        cams[idx] = cam
    return cams


def find_all_predictions(path) -> dict[int, str]:
    return {int(p.stem): str(p) for p in Path(path).glob("**/*.pfm")}


def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape(
        (4, 4)
    )
    intrinsics = np.fromstring(
        " ".join(lines[7:10]), dtype=np.float32, sep=" "
    ).reshape((3, 3))

    return intrinsics, extrinsics


def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.0
    return np_img


def read_mask(filename):
    return read_img(filename) > 0.5


def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data


def reproject_with_depth(
    depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src
):
    batch, height, width = depth_ref.shape
    ## step1. project reference pixels to the source view
    # reference view x, y
    y_ref, x_ref = torch.meshgrid(
        torch.arange(0, height).to(depth_ref.device),
        torch.arange(0, width).to(depth_ref.device),
    )
    x_ref = x_ref.unsqueeze(0).repeat(batch, 1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch, 1, 1)
    x_ref, y_ref = x_ref.reshape(batch, -1), y_ref.reshape(batch, -1)
    # reference 3D space

    A = torch.inverse(intrinsics_ref)
    B = torch.stack(
        (x_ref, y_ref, torch.ones_like(x_ref).to(x_ref.device)), dim=1
    ) * depth_ref.reshape(batch, 1, -1)
    xyz_ref = torch.matmul(A, B)

    # source 3D space
    xyz_src = torch.matmul(
        torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)),
        torch.cat(
            (xyz_ref, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1
        ),
    )[:, :3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:, :2] / K_xyz_src[:, 2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[:, 0].reshape([batch, height, width]).float()
    y_src = xy_src[:, 1].reshape([batch, height, width]).float()

    # print(x_src, y_src)
    sampled_depth_src = bilinear_sampler(
        depth_src.view(batch, 1, height, width),
        torch.stack((x_src, y_src), dim=-1).view(batch, height, width, 2),
    )

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(
        torch.inverse(intrinsics_src),
        torch.cat((xy_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1)
        * sampled_depth_src.reshape(batch, 1, -1),
    )
    # reference 3D space
    xyz_reprojected = torch.matmul(
        torch.matmul(extrinsics_ref, torch.inverse(extrinsics_src)),
        torch.cat(
            (xyz_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1
        ),
    )[:, :3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2].reshape([batch, height, width]).float()
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:, :2] / K_xyz_reprojected[:, 2:3]
    x_reprojected = xy_reprojected[:, 0].reshape([batch, height, width]).float()
    y_reprojected = xy_reprojected[:, 1].reshape([batch, height, width]).float()

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(
    depth_ref,
    intrinsics_ref,
    extrinsics_ref,
    depth_src,
    intrinsics_src,
    extrinsics_src,
    thre1=4.4,
    thre2=1430.0,
):
    batch, height, width = depth_ref.shape
    y_ref, x_ref = torch.meshgrid(
        torch.arange(0, height).to(depth_ref.device),
        torch.arange(0, width).to(depth_ref.device),
    )
    x_ref = x_ref.unsqueeze(0).repeat(batch, 1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch, 1, 1)
    inputs = [
        depth_ref,
        intrinsics_ref,
        extrinsics_ref,
        depth_src,
        intrinsics_src,
        extrinsics_src,
    ]
    outputs = reproject_with_depth(*inputs)
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = outputs
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    masks = []
    for i in range(2, batch + 1):
        mask = torch.logical_and(dist < i / thre1, relative_depth_diff < i / thre2)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff


def pad(img, h, w, mode="constant"):
    origh, origw = img.shape[:2]
    hpad = (0, 0)
    if h > origh:
        hpad = ((h - origh) // 2, int(np.ceil((h - origh) / 2)))
    wpad = (0, 0)
    if w > origw:
        wpad = ((w - origw) // 2, int(np.ceil((w - origw) / 2)))
    return np.pad(img, [hpad, wpad, (0, 0)], mode=mode)


def center_crop(img, h, w):
    origh, origw = img.shape[:2]
    if h < origh:
        p = (origh - h) // 2
        img = img[p : h + p]
    if w < origw:
        p = (origw - w) // 2
        img = img[:, p : w + p]
    return img


def adjust_intrinsics(depth, intrinsics, h, w) -> np.ndarray:
    origh, origw = depth.shape[:2]
    intrins = intrinsics.copy()
    if h > origh:
        intrins[1, -1] += (h - origh) // 2
    if w > origw:
        intrins[0, -1] += (w - origw) // 2
    return intrins


# MAIN

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fusion of depth maps in a Point Cloud"
    )
    parser.add_argument("scan_folder", type=str, help="Scan Folder")
    parser.add_argument(
        "pred_folder", type=str, help="Folder containing predictions in .pfm format"
    )
    parser.add_argument("out", type=str, help="Path for the output ply file")
    parser.add_argument(
        "--glb", type=float, default=0.25, help="global geo mask threshold"
    )

    args = parser.parse_args()
    pcd_fusion(args.scan_folder, args.pred_folder, args.out, args.glb)
