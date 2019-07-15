import json

import numpy as np
import pytest
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

from h5py_like import Mode, FileMixin
from h5py_like.test_utils import FileTestBase, DatasetTestBase, GroupTestBase, ModeTestBase
from pyn5 import File

ds_kwargs = deepcopy(DatasetTestBase.dataset_kwargs)
ds_kwargs["chunks"] = (5, 5, 5)


class TestFile(FileTestBase):
    dataset_kwargs = ds_kwargs
    pass


class TestGroup(GroupTestBase):
    dataset_kwargs = ds_kwargs
    pass


class TestDataset(DatasetTestBase):
    dataset_kwargs = ds_kwargs

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


def test_data_ordering(file_, h5_file, random):
    data = random.random_sample((11, 12, 13))
    ds_kwargs = {"shape": data.shape, "dtype": data.dtype, "chunks": (5, 5, 5)}

    for f in (file_, h5_file):
        n5_ds = f.create_dataset("ds", **ds_kwargs)
        n5_ds[:] = data

    assert np.allclose(file_["ds"][:], h5_file["ds"][:])


def test_created_dirs(file_):
    shape = (10, 20)
    data = np.ones(shape)

    ds = file_.create_dataset("ds", data=data, chunks=(10, 10))

    created = {
        str(path.relative_to(ds._path))
        for path in ds._path.glob('**/*')
        if path.suffix != ".json"
    }

    assert created == {"0", "1", "0/0", "1/0"}

    attrs = ds.attrs._read_attributes()

    assert list(ds.shape) == attrs["dimensions"][::-1]
