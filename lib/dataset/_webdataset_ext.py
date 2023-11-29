"""
Extensions to WebDataset to easily load sequences from standard
sequential datasets easily saving time and memory
"""

import os
import tarfile
import warnings
from typing import IO, Any, Iterable, Iterator, Optional, Tuple, cast
from torchdata.datapipes.iter import IterDataPipe
from collections import deque
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter.util.webdataset import pathsplit
from torchdata.datapipes.utils.common import (
    validate_pathname_binary_tuple,
    StreamWrapper,
)
import torchvision.transforms.functional as F
from io import BufferedIOBase

__all__ = [
    "LoadFromTarWithArchive",
    "WebDatasetWithArchive",
    "LoadPrevs",
]


@functional_datapipe("load_from_tar_with_archive")
class LoadFromTarWithArchive(IterDataPipe[Tuple[str, tarfile.TarFile, BufferedIOBase]]):
    def __init__(
        self,
        datapipe: Iterable[Tuple[str, BufferedIOBase]],
        mode: str = "r:*",
        length: int = -1,
    ):
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        self.mode: str = mode
        self.length: int = length

    def __iter__(self) -> Iterator[Tuple[str, tarfile.TarFile, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                # typing.cast is used here to silence mypy's type checker
                tar = tarfile.open(
                    fileobj=cast(Optional[IO[bytes]], data_stream), mode=self.mode
                )
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        warnings.warn(
                            f"failed to extract file {tarinfo.name} from source tarfile {pathname}"
                        )
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(
                        os.path.join(pathname, tarinfo.name)
                    )
                    yield inner_pathname, tar, StreamWrapper(extracted_fobj)  # type: ignore[misc]
            except Exception as e:
                warnings.warn(
                    f"Unable to extract files from corrupted tarfile stream {pathname} due to: {e}, abort!"
                )
                raise e

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length


@functional_datapipe("webdataset_with_archive")
class WebDatasetWithArchive(IterDataPipe[dict]):
    """See WebDatasetIterDataPipe"""

    def __init__(self, source_datapipe: IterDataPipe[list[dict | list]]) -> None:
        self.source_datapipe: IterDataPipe[list[dict | list]] = source_datapipe

    def __iter__(self) -> Iterator[dict]:
        sample: dict[str, Any] = {}
        current = ""
        for path, archive, data in self.source_datapipe:
            assert isinstance(path, str), path
            prefix, suffix = pathsplit(path)
            if suffix == "":
                # files with empty suffixes can be used for metadata
                # they cannot be used for data since they wouldn't have a key
                continue
            if prefix != current:
                if current != "":
                    yield sample
                sample = {}
                current = prefix
                sample["__key__"] = current
                sample["__archive__"] = archive
            sample[suffix] = data
            sample[f"__path__{suffix}"] = path
        if sample != {}:
            yield sample

    def __len__(self) -> int:
        return len(self.source_datapipe)


class _NotAutoCloseStreamWrapper(StreamWrapper):
    def close(self, *args, **kwargs):
        # We need to ignore autoclosing since it interferes with
        # StreamWrapper sharing that is what it allows blazing
        # fast loading of long sequences
        pass

    def _close(self, *args, **kwargs):
        # if (
        #     hasattr(StreamWrapper, "debug_unclosed_streams")
        #     and StreamWrapper.debug_unclosed_streams
        # ):
        #     del StreamWrapper.session_streams[self]
        if hasattr(self, "parent_stream") and self.parent_stream is not None:
            self.parent_stream.child_counter -= 1
            if (
                not self.parent_stream.child_counter
                and self.parent_stream.close_on_last_child
            ):
                self.parent_stream.close()
        try:
            self.file_obj.close(*args, **kwargs)
        except AttributeError:
            pass
        self.closed = True

    def __del__(self):
        # still we want to close when out of scope
        if not self.closed:
            self._close()


@functional_datapipe("load_prevs")
class LoadPrevs(IterDataPipe[dict]):
    def __init__(
        self,
        source_datapipe: IterDataPipe[dict],
        prevs: int = 0,
    ):
        self.source_datapipe = source_datapipe
        self.load_prevs = prevs

    def prev(self, name, idx):
        _, name, ext = name.split(".")
        return "." + name + f"_prev_{idx}." + ext

    def _copy_sample(self, sample: dict):
        out = {
            k: _NotAutoCloseStreamWrapper(
                sample["__archive__"].extractfile(
                    os.path.split(sample[f"__path__{k}"])[1]
                )
            )
            for k in sample
            if not k.startswith("__")
        } | {k: v for k, v in sample.items() if k.startswith("__")}
        out["__key__"] = int(sample["__key__"].split("/")[-1])
        return out

    def __iter__(self) -> Iterator[dict]:
        queue = deque(maxlen=self.load_prevs + 1)
        for sample in self.source_datapipe:
            key = int(sample["__key__"].split("/")[-1])
            if len(queue) > 0 and key <= queue[0]["__key__"]:
                queue.clear()

            queue.appendleft(self._copy_sample(sample))
            if len(queue) == self.load_prevs + 1:
                out = queue[0]
                for idx in range(1, self.load_prevs + 1):
                    out = out | {
                        self.prev(k, idx - 1): v
                        for k, v in queue[idx].items()
                        if not k.startswith("__")
                    }
                yield {k: v for k, v in out.items() if not k.startswith("__")}
