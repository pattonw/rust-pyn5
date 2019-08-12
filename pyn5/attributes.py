import errno
import json
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps

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
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def restrict_metadata(fn):
    """Decorator for AttributeManager methods which prevents mutation of N5 metadata"""
    @wraps(fn)
    def wrapped(obj: "AttributeManager", key, *args, **kwargs):
        if obj._is_dataset() and key in obj._dataset_keys:
            raise RuntimeError(f"N5 metadata (key '{key}') cannot be mutated")
        return fn(obj, key, *args, **kwargs)
    return mutation(wrapped)


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
        self._has_dataset_keys_ = None
        super().__init__(mode)

    @classmethod
    def from_container(cls, container: H5ObjectLike) -> "AttributeManager":
        """
        Create AttributeManager for a File, Group or Dataset.

        :param container: File, Group or Dataset to which the attributes belong.
        :return: AttributeManager instance
        """
        return cls(container._path, container.mode)

    @restrict_metadata
    def __setitem__(self, k, v) -> None:
        with self._open_attributes(True) as attrs:
            attrs[k] = v

    @restrict_metadata
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

    def _is_dataset(self) -> bool:
        if self._has_dataset_keys_ is None:
            try:
                with open(self._path, "r") as f:
                    self._has_dataset_keys_ = self._dataset_keys.issubset(json.load(f))
            except ValueError:
                self._has_dataset_keys_ = False
            except IOError as e:
                if e.errno == errno.ENOENT:
                    self._has_dataset_keys_ = False
                else:
                    raise
        return self._has_dataset_keys_

    @contextmanager
    def _open_attributes(self, write: bool = False) -> Dict[str, Any]:
        """Return attributes as a context manager.

        N5 metadata keys are stripped from the dict.

        :param write: Whether to write changes to the attributes dict.
        :return: attributes as a dict
        """
        attributes = self._read_attributes()

        if self._is_dataset():
            hidden_attrs = {k: attributes.pop(k) for k in self._dataset_keys}
        else:
            hidden_attrs = dict()

        yield attributes
        if write:
            hidden_attrs.update(attributes)
            self._write_attributes(hidden_attrs)

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
