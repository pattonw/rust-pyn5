from copy import copy

from typing import Union, Tuple, Optional, Any

import numpy as np

from h5py_like import DatasetBase, AttributeManagerBase, mutation
from h5py_like.shape_utils import thread_read_fn, thread_write_fn
from pyn5.attributes import AttributeManager
from .pyn5 import (
    DatasetUINT8,
    DatasetUINT16,
    DatasetUINT32,
    DatasetUINT64,
    DatasetINT8,
    DatasetINT16,
    DatasetINT32,
    DatasetINT64,
    DatasetFLOAT32,
    DatasetFLOAT64,
)


dataset_types = {
    np.dtype("uint8"): DatasetUINT8,
    np.dtype("uint16"): DatasetUINT16,
    np.dtype("uint32"): DatasetUINT32,
    np.dtype("uint64"): DatasetUINT64,
    np.dtype("int8"): DatasetINT8,
    np.dtype("int16"): DatasetINT16,
    np.dtype("int32"): DatasetINT32,
    np.dtype("int64"): DatasetINT64,
    np.dtype("float32"): DatasetFLOAT32,
    np.dtype("float64"): DatasetFLOAT64,
}


class Dataset(DatasetBase):
    threads = None

    def __init__(self, basename: str, parent: "Group"):  # noqa would need circular imports
        """

        :param basename: basename of the dataset
        :param parent: group to which the dataset belongs
        """
        super().__init__(basename, parent)
        self._path = self.parent._path / self.basename
        self._attrs = AttributeManager.from_container(self)
        self.threads = copy(self.threads)

        attrs = self._attrs._read_attributes()

        try:
            self._shape = tuple(attrs["dimensions"][::-1])
            self._dtype = np.dtype(attrs["dataType"].lower())
            self._chunks = tuple(attrs["blockSize"][::-1])
        except KeyError:
            raise ValueError(
                f"Not a dataset (missing metadata key): {self._path}"
            )

        self._impl = dataset_types[self.dtype](
            str(self.file._path),
            self.name[1:],
            True,  # raise error if dataset does not exist on disk
        )

    @property
    def dims(self):
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def maxshape(self) -> Tuple[int, ...]:
        return self.shape

    @property
    def fillvalue(self) -> Any:
        return 0

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        return self._chunks

    def resize(self, size: Union[int, Tuple[int, ...]], axis: Optional[int] = None):
        raise NotImplementedError()

    def __getitem__(self, args) -> np.ndarray:
        def inner_fn(translation, dimensions):
            return self._impl.read_ndarray(
                translation[::-1], dimensions[::-1]
            ).transpose()

        if self.threads:
            def fn(translation, dimensions):
                return thread_read_fn(
                    translation,
                    dimensions,
                    self.chunks,
                    self.shape,
                    inner_fn,
                    self.threads
                )
        else:
            fn = inner_fn

        return self._getitem(args, fn, self._astype)

    @mutation
    def __setitem__(self, args, val):
        def inner_fn(offset, arr):
            return self._impl.write_ndarray(
                offset[::-1], arr.transpose(), self.fillvalue
            )

        if self.threads:
            def fn(offset, arr):
                return thread_write_fn(
                    offset,
                    arr,
                    self.chunks,
                    self.shape,
                    inner_fn,
                    self.threads
                )
        else:
            fn = inner_fn

        return self._setitem(args, val, fn)

    @property
    def attrs(self) -> AttributeManagerBase:
        return self._attrs
