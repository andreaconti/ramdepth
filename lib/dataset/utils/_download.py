"""
Utilities to download data from GitHub Releases
"""

from urllib.request import urlopen, Request
from urllib.error import HTTPError
from zipfile import ZipFile
import tempfile
import torch
from pathlib import Path
import os
from tqdm import tqdm
import hashlib
import shutil

__all__ = [
    "github_download_release_asset",
    "github_download_unzip_assets",
    "download_url_to_file",
    "root_data",
]


def root_data(name: str):
    base_dir = Path(torch.hub.get_dir())
    root = base_dir / f"ramdepth/data/{name}"
    return root


def github_download_unzip_assets(
    owner: str,
    repo: str,
    asset_ids: list[str],
    dst: str | Path,
):
    try:
        Path(dst).mkdir(exist_ok=True, parents=True)
        tmpdir = tempfile.mkdtemp()
        for asset_id in asset_ids:
            tmpfile = os.path.join(tmpdir, asset_id)
            github_download_release_asset(owner, repo, asset_id, tmpfile)
            with ZipFile(tmpfile) as f:
                f.extractall(dst)
    finally:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


def github_download_release_asset(
    owner: str,
    repo: str,
    asset_id: str,
    dst: str | Path,
):
    headers = {
        "Accept": "application/octet-stream",
    }
    if token := os.environ.get("GITHUB_TOKEN", None):
        headers["Authorization"] = f"Bearer {token}"

    try:
        download_url_to_file(
            f"https://api.github.com/repos/{owner}/{repo}/releases/assets/{asset_id}",
            dst,
            headers=headers,
        )
    except HTTPError as e:
        if e.code == 404 and "Authorization" not in headers:
            raise RuntimeError(
                "File not found, maybe missing GITHUB_TOKEN env variable?"
            )
        raise e


def download_url_to_file(
    url,
    dst,
    *,
    hash_prefix=None,
    progress=True,
    headers=None,
):
    # find file size
    file_size = None
    req = Request(url, headers={} if not headers else headers)
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(
            total=file_size,
            disable=not progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    'invalid hash value (expected "{}", got "{}")'.format(
                        hash_prefix, digest
                    )
                )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
