import json
import hashlib
from typing import NamedTuple, Optional, Tuple

try:
    import z5py
except ImportError:
    z5py = None


def blocks_in(dpath):
    return {
        str(path.relative_to(dpath))
        for path in dpath.glob('**/*')
        if path.suffix != ".json"
    }


def iter_block_paths(dpath):
    for fpath in sorted(
        fpath for fpath in dpath.glob('**/*')
        if fpath.is_file() and fpath.suffix != ".json"
    ):
        yield fpath


def blocks_hash(dpath):
    md5 = hashlib.md5()
    for fpath in iter_block_paths(dpath):
        md5.update(str(fpath.relative_to(dpath)).encode())
        md5.update(fpath.read_bytes())

    return md5.hexdigest()


def attrs_in(dpath):
    return json.loads((dpath / "attributes.json").read_text())


class BlockContents(NamedTuple):
    mode: int
    ndim: int
    shape: Tuple[int, ...]
    num_elem: Optional[int]
    compressed_data: bytes

    @classmethod
    def from_block(cls, fpath):
        with open(fpath, 'rb') as f:
            mode = int.from_bytes(f.read(2), "big", signed=False)
            ndim = int.from_bytes(f.read(2), "big", signed=False)
            shape = tuple(
                int.from_bytes(f.read(4), "big", signed=False)
                for _ in range(ndim)
            )
            if mode:
                num_elem = int.from_bytes(f.read(4), "big", signed=False)
            else:
                num_elem = None
            data = f.read()

        return BlockContents(mode, ndim, shape, num_elem, data)
