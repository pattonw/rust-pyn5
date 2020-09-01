# -*- coding: utf-8 -*-

"""Top-level package for pyn5."""

__author__ = """William Hunter Patton"""
__email__ = "pattonw@hhmi.org"
__version__ = "1.1.1"

from h5py_like import Mode

from .python_wrappers import open, read, write
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
    create_dataset,
)
from .attributes import AttributeManager
from .dataset import Dataset
from .file_group import File, Group
from .common import CompressionType

__all__ = [
    "open",
    "read",
    "write",
    "create_dataset",
    "DatasetUINT8",
    "DatasetUINT16",
    "DatasetUINT32",
    "DatasetUINT64",
    "DatasetINT8",
    "DatasetINT16",
    "DatasetINT32",
    "DatasetINT64",
    "DatasetFLOAT32",
    "DatasetFLOAT64",
    "File",
    "Group",
    "Dataset",
    "AttributeManager",
    "CompressionType",
    "Mode",
]
