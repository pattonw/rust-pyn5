from copy import deepcopy

import pytest
import numpy as np

import pyn5

try:
    import h5py
except ImportError:
    h5py = None

from .common import z5py

DS_SIZE = (10, 10, 10)
BLOCKSIZE = (2, 2, 2)

INT_DTYPES = ["UINT8", "UINT16", "UINT32", "UINT64", "INT8", "INT16", "INT32", "INT64"]

FLOAT_DTYPES = ["FLOAT32", "FLOAT64"]


@pytest.fixture
def random():
    return np.random.RandomState(1991)


@pytest.fixture(params=INT_DTYPES + FLOAT_DTYPES)
def ds_dtype(request, tmp_path):
    dtype = request.param
    n5_path = str(tmp_path / "test.n5")
    ds_name = "ds" + dtype

    pyn5.create_dataset(n5_path, ds_name, DS_SIZE, BLOCKSIZE, dtype)
    yield pyn5.open(n5_path, ds_name, dtype, False), np.dtype(dtype.lower())


@pytest.fixture(params=[
    {"type": "raw"},
    {"type": "bzip2", "blockSize": 5},
    {"type": "gzip", "level": 5},
    # {"type": "lz4", "blockSize": 32768},
    # {"type": "xz", "preset": 3},
], ids=lambda d: d.get("type", "raw"))
def compression_dict(request):
    yield deepcopy(request.param)


@pytest.fixture
def compression_name_opt(compression_dict):
    name = compression_dict.pop("type")
    arg = list(compression_dict.values()).pop() if compression_dict else None
    yield name, arg


@pytest.fixture
def file_(tmp_path):
    f = pyn5.File(tmp_path / "test.n5")
    yield f


@pytest.fixture
def h5_file(tmp_path):
    if not h5py:
        pytest.skip("h5py not installed")

    with h5py.File(tmp_path / "test.hdf5") as f:
        yield f


@pytest.fixture
def z5_file(tmp_path):
    if not z5py:
        pytest.skip("z5py not installed")

    yield z5py.N5File(tmp_path / "test_z5.n5")
