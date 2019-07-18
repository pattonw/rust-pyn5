import json

import numpy as np
import pytest
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

from h5py_like import Mode, FileMixin
from h5py_like.test_utils import (
    FileTestBase, ThreadedDatasetTestBase, GroupTestBase, ModeTestBase,
)
from pyn5 import File

from .common import blocks_hash
from .common import blocks_in, attrs_in

ds_kwargs = deepcopy(ThreadedDatasetTestBase.dataset_kwargs)
ds_kwargs["chunks"] = (5, 5, 5)


class TestFile(FileTestBase):
    dataset_kwargs = ds_kwargs
    pass


class TestGroup(GroupTestBase):
    dataset_kwargs = ds_kwargs
    pass


class TestDataset(ThreadedDatasetTestBase):
    dataset_kwargs = deepcopy(ThreadedDatasetTestBase.dataset_kwargs)

    def test_has_metadata(self, file_):
        ds = self.dataset(file_)
        with open(ds.attrs._path) as f:
            attrs = json.load(f)
        for key in ds.attrs._dataset_keys:
            assert key in attrs

    def test_no_return_metadata(self, file_):
        ds = self.dataset(file_)

        for key in ds.attrs._dataset_keys:
            assert key not in ds.attrs
            assert key not in dict(ds.attrs)

    def test_no_mutate_metadata(self, file_):
        ds = self.dataset(file_)

        for key in ds.attrs._dataset_keys:
            with pytest.raises(RuntimeError):
                ds.attrs[key] = "not a datatype"

            with pytest.raises(RuntimeError):
                del ds.attrs[key]


class TestMode(ModeTestBase):
    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        try:
            shutil.rmtree(self.tmp_dir)
        except FileNotFoundError:
            pass

    def factory(self, mode: Mode) -> FileMixin:
        fpath = self.tmp_dir / "test.n5"
        return File(fpath, mode)


def test_data_ordering(file_, h5_file):
    shape = (5, 10, 15)
    data = np.arange(np.product(shape)).reshape(shape)
    ds_kwargs = {"shape": data.shape, "dtype": data.dtype, "chunks": (4, 4, 4)}

    for f in (file_, h5_file):
        n5_ds = f.create_dataset("ds", **ds_kwargs)
        n5_ds[:] = data

    assert np.allclose(file_["ds"][:], h5_file["ds"][:])


def test_created_dirs(file_):
    shape = (10, 20)
    data = np.ones(shape)

    ds = file_.create_dataset("ds", data=data, chunks=(10, 10))

    assert blocks_in(ds._path) == {"0", "1", "0/0", "1/0"}

    attrs = ds.attrs._read_attributes()

    assert list(ds.shape) == attrs["dimensions"][::-1]


def test_vs_z5(file_, z5_file):
    """Check same as z5"""
    z5_path = Path(z5_file.path)
    shape = (10, 20)
    data = np.arange(np.product(shape)).reshape(shape)

    file_.create_dataset("ds", data=data, chunks=(6, 7))
    z5_file.create_dataset("ds", data=data, chunks=(6, 7), compression="gzip", level=-1)

    assert np.allclose(file_["ds"][:], z5_file["ds"][:])
    assert blocks_in(file_["ds"]._path) == blocks_in(z5_path / "ds")

    attrs = attrs_in(file_["ds"]._path)
    z5_attrs = attrs_in(z5_path / "ds")
    for key in ("blockSize", "dimensions", "dataType", "compression"):
        assert attrs[key] == z5_attrs[key]


def test_vs_z5_hash(file_, z5_file):
    """Check identical block contents to z5"""
    z5_path = Path(z5_file.path)
    shape = (10, 20)
    data = np.arange(np.product(shape)).reshape(shape)

    file_.create_dataset("ds", data=data, chunks=(6, 7), compression="raw")
    z5_file.create_dataset("ds", data=data, chunks=(6, 7), compression="raw")

    assert blocks_hash(file_._path) == blocks_hash(z5_path)


def test_compression_opts(file_, compression_name_opt):
    compression, opt = compression_name_opt

    shape = (10, 20)
    data = np.arange(np.product(shape)).reshape(shape)

    ds = file_.create_dataset(
        "ds", data=data, chunks=(10, 10), compression=compression, compression_opts=opt
    )

    compression_dict = ds.attrs._read_attributes()["compression"]
    assert compression_dict.pop("type") == compression
    if opt is not None:
        assert list(compression_dict.values()).pop() == opt
