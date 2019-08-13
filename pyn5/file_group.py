import json

import shutil
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np

from h5py_like import GroupBase, FileMixin, Mode, mutation
from h5py_like.common import Name
from h5py_like.base import H5ObjectLike
from h5py_like.shape_utils import guess_chunks

from pyn5 import Dataset
from .common import compression_args
from .attributes import AttributeManager
from .pyn5 import create_dataset

N5_VERSION = "2.0.2"
N5_VERSION_INFO = tuple(int(i) for i in N5_VERSION.split('.'))


class Group(GroupBase):
    def __init__(self, basename: str, parent: "Group"):
        """

        :param basename: basename of the group
        :param parent: group to which the group belongs
        """
        super().__init__(basename, parent)
        self._path = self.parent._path / self.basename

        self._attrs = AttributeManager.from_container(self)

    def _create_child_group(self, name) -> GroupBase:
        dpath = self._path / name

        try:
            obj = self._get_child(name)
        except KeyError:
            pass
        else:
            if isinstance(obj, Group):
                raise FileExistsError(f"Group already exists at {dpath}")
            else:
                raise TypeError(f"Dataset found at {dpath}")

        dpath.mkdir()
        return Group(name, self)

    def _create_child_dataset(
        self, name, shape=None, dtype=None, data=None, chunks=None,
        compression=None, compression_opts=None, **kwds
    ):
        for key in kwds:
            warnings.warn(
                f"pyn5 does not implement '{key}' argument for create_dataset; it will be ignored"
            )

        if data is not None:
            data = np.asarray(data, dtype=dtype)
            dtype = data.dtype
            shape = data.shape

        dtype = np.dtype(dtype)

        if chunks is None:
            warnings.warn(
                "chunks not set: entire dataset will be a single chunk. "
                "This will be slow and inefficient. "
                "Set chunks=True to guess reasonable chunk sizes."
            )
            chunks = shape
        elif chunks is True:
            chunks = guess_chunks(shape, dtype.itemsize)

        dpath = self._path / name

        try:
            obj = self._get_child(name)
        except KeyError:
            pass
        else:
            if isinstance(obj, Dataset):
                raise TypeError(f"Dataset already exists at {dpath}")
            elif isinstance(obj, Group):
                raise FileExistsError(f"Group found at {dpath}")

        if compression:
            try:
                opt_name = compression_args[compression]
            except KeyError:
                raise ValueError(
                    f"Unknown compression type '{compression}': "
                    f"use one of {sorted(compression_args)}"
                )

            compression_dict = {"type": str(compression)}
            if compression_opts is not None:
                compression_dict[opt_name] = compression_opts

            compression_str = json.dumps(compression_dict)
        else:
            compression_str = None

        file_path = str(self.file.filename)
        create_dataset(
            file_path,
            str(Name(self.name) / name)[1:],
            list(shape)[::-1],
            list(chunks)[::-1],
            dtype.name.upper(),
            compression_str,
        )

        ds = Dataset(name, self)
        if data is not None:
            ds[...] = data
        return ds

    def _get_child(self, name) -> H5ObjectLike:
        dpath = self._path / name
        if not dpath.is_dir():
            raise KeyError()
        attrs = AttributeManager(dpath)
        if attrs._is_dataset():
            return Dataset(name, self)
        else:
            return Group(name, self)

    @mutation
    def __setitem__(self, name, obj):
        """Not implemented"""
        raise NotImplementedError()

    def copy(
        self,
        source,
        dest,
        name=None,
        shallow=False,
        expand_soft=False,
        expand_external=False,
        expand_refs=False,
        without_attrs=False,
    ):
        """Not implemented"""
        raise NotImplementedError()

    @property
    def attrs(self) -> AttributeManager:
        return self._attrs

    @mutation
    def __delitem__(self, v) -> None:
        shutil.rmtree(self[v]._path)

    def __len__(self) -> int:
        counter = 0
        for _ in self:
            counter += 1
        return counter

    def __iter__(self) -> Iterator:
        for path in self._path.iterdir():
            if path.is_dir():
                yield path.name


class File(FileMixin, Group):
    def __init__(self, name, mode=Mode.READ_WRITE_CREATE):
        super().__init__(name, mode)
        self._require_dir(self.filename)
        self._path = self.filename
        self._attrs = AttributeManager.from_container(self)

    @mutation
    def __setitem__(self, name, obj):
        """Not implemented"""
        raise NotImplementedError()

    def copy(
        self,
        source,
        dest,
        name=None,
        shallow=False,
        expand_soft=False,
        expand_external=False,
        expand_refs=False,
        without_attrs=False,
    ):
        """Not implemented"""
        raise NotImplementedError()

    def _require_dir(self, dpath: Path):
        if dpath.is_file():
            raise FileExistsError("File found at desired location of directory")
        created = False
        if dpath.is_dir():
            if self.mode == Mode.CREATE:
                raise FileExistsError()
            elif self.mode == Mode.CREATE_TRUNCATE:
                shutil.rmtree(dpath)
                dpath.mkdir()
                created = True
        else:
            if self.mode in (Mode.READ_ONLY, Mode.READ_WRITE):
                raise FileNotFoundError()
            else:
                dpath.mkdir(parents=True)
                created = True

        attrs = AttributeManager(dpath, self.mode)
        if created:
            attrs["n5"] = N5_VERSION
        else:
            version = attrs.get("n5")
            if not version:
                raise ValueError(f"No N5 version found in {attrs._path}")

            version_info = tuple(int(i) for i in version.split('.'))

            if version_info[0] != N5_VERSION_INFO[0]:
                raise ValueError(f"Expected N5 version '{N5_VERSION}', got {version}")
            elif version_info[1] != N5_VERSION_INFO[1]:
                warnings.warn(f"Expected N5 version '{N5_VERSION}', got {version};"
                              f" trying to open anyway")

        return created
