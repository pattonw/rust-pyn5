from __future__ import annotations
import errno
import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from typing import Iterator, Any, Dict

import numpy as np

from h5py_like import AttributeManagerBase, mutation, Mode
from h5py_like.base import H5ObjectLike


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder object which converts numpy arrays to lists."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class AttributeManager(AttributeManagerBase):
    """Object which reads and writes group attributes as JSON.

    The ``_encoder`` member variable (default ``NumpyEncoder``) can be set on
    the class or the instance to change how attributes are serialised.

    The ``_dump_kwargs`` member variable is passed as kwargs to ``json.dump`` on write.
    By default, it is an empty dict.
    New instances make a deep copy of the class variable.
    """

    _dataset_keys = {"dimensions", "blockSize", "dataType", "compression"}
    _encoder = NumpyEncoder
    _dump_kwargs = dict()

    def __init__(self, dpath: Path, mode=Mode.default()):
        """
        :param dpath: Path of the directory in which the attributes file resides.
        :param mode: Mode
        """
        self._path = Path(dpath) / "attributes.json"
        self._dump_kwargs = deepcopy(self._dump_kwargs)
        super().__init__(mode)

    @classmethod
    def from_parent(cls, parent: H5ObjectLike) -> AttributeManager:
        """
        Create AttributeManager for a File, Group or Dataset.

        :param parent: File, Group or Dataset to which the attributes belong.
        :return: AttributeManager instance
        """
        return cls(parent._path, parent.mode)

    @mutation
    def __setitem__(self, k, v) -> None:
        with self._open_attributes(True) as attrs:
            attrs[k] = v

    @mutation
    def __delitem__(self, v) -> None:
        with self._open_attributes(True) as attrs:
            del attrs[v]

    def __getitem__(self, k):
        with self._open_attributes() as attrs:
            return attrs[k]

    def __len__(self) -> int:
        with self._open_attributes() as attrs:
            return len(attrs)

    def __iter__(self) -> Iterator:
        yield from self.keys()

    def keys(self):
        with self._open_attributes() as attrs:
            return attrs.keys()

    def values(self):
        """Mutations are not written back to the attributes file"""
        with self._open_attributes() as attrs:
            return attrs.values()

    def items(self):
        """Mutations are not written back to the attributes file"""
        with self._open_attributes() as attrs:
            return attrs.items()

    def __contains__(self, item):
        with self._open_attributes() as attrs:
            return item in attrs

    def _is_dataset(self):
        with self._open_attributes() as attrs:
            return self._dataset_keys.issubset(attrs)

    @contextmanager
    def _open_attributes(self, write: bool = False) -> Dict[str, Any]:
        """Return attributes as a context manager.

        :param write: Whether to write changes to the attributes dict.
        :return: attributes as a dict (including N5 metadata)
        """
        attributes = self._read_attributes()
        yield attributes
        if write:
            self._write_attributes(attributes)

    def _read_attributes(self):
        """Return attributes or an empty dict if they do not exist"""
        try:
            with open(self._path, "r") as f:
                attributes = json.load(f)
        except ValueError:
            attributes = {}
        except IOError as e:
            if e.errno == errno.ENOENT:
                attributes = {}
            else:
                raise

        return attributes

    def _write_attributes(self, attrs):
        """Write dict to attributes file, using AttributeManager's encoder and kwargs."""
        with open(self._path, "w") as f:
            json.dump(attrs, f, cls=self._encoder, **self._dump_kwargs)
