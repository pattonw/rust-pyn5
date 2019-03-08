# -*- coding: utf-8 -*-

"""Main module."""
import json
from pathlib import Path
import logging
from typing import Tuple
import numpy as np

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


def open(root_path: str, dataset: str, dtype: str = "", read_only=True):
    """
    Returns a Dataset of the corresponding dtype. Leave dtype blank to return
    the Dataset with dtype as shown in the attributes.json file
    """

    # Check the attributes file:
    attributes_file = Path(root_path, dataset, "attributes.json")
    if attributes_file.exists():
        with attributes_file.open("r") as f:
            attributes = json.load(f)
        expected_dtype = attributes.get("dataType", None)
        if expected_dtype is not None:
            if dtype == "":
                # Use the expected dtype
                return open(root_path, dataset, expected_dtype.upper(), read_only)
            elif dtype != expected_dtype.upper():
                # When in doubt use the user specified dtype
                logging.warning(
                    "Given dtype {} does not match dtype ({}) in attributes.json".format(
                        dtype, expected_dtype.upper()
                    )
                )

    if dtype == "UINT8":
        return DatasetUINT8(root_path, dataset, read_only)
    elif dtype == "UINT16":
        return DatasetUINT16(root_path, dataset, read_only)
    elif dtype == "UINT32":
        return DatasetUINT32(root_path, dataset, read_only)
    elif dtype == "UINT64":
        return DatasetUINT64(root_path, dataset, read_only)
    elif dtype == "INT8":
        return DatasetINT8(root_path, dataset, read_only)
    elif dtype == "INT16":
        return DatasetINT16(root_path, dataset, read_only)
    elif dtype == "INT32":
        return DatasetINT32(root_path, dataset, read_only)
    elif dtype == "INT64":
        return DatasetINT64(root_path, dataset, read_only)
    elif dtype == "FLOAT32":
        return DatasetFLOAT32(root_path, dataset, read_only)
    elif dtype == "FLOAT64":
        return DatasetFLOAT64(root_path, dataset, read_only)
    else:
        raise ValueError(
            "Given dtype {} is not supported. Please choose from ({})".format(
                dtype,
                (
                    "UINT8",
                    "UINT16",
                    "UINT32",
                    "UINT64",
                    "INT8",
                    "INT16",
                    "INT32",
                    "INT64",
                    "FLOAT32",
                    "FLOAT64",
                ),
            )
        )


def read(dataset, bounds: Tuple[np.ndarray, np.ndarray], dtype: type = int):
    """
    Temporary hacky method until dataset.read_ndarray returns np.ndarray

    Note: passing in dtype is necessary since numpy arrays are float by default.
          dataset.get_data_type() could be implemented, but a better solution would
          be to have dataset.read_ndarray return a numpy array. 
    """
    bounds = (bounds[0].astype(int), bounds[1].astype(int))
    return (
        np.array(dataset.read_ndarray(list(bounds[0]), list(bounds[1] - bounds[0])))
        .reshape(list(bounds[1] - bounds[0]))
        .transpose([2, 1, 0])
        .astype(dtype)
    )


def write(
    dataset,
    input_bounds: Tuple[np.ndarray, np.ndarray],
    input_data: np.ndarray,
    dtype=int,
):
    """
    Temporary hacky method until dataset.write_ndarray is implemented in rust-n5
    and the PyO3 wrapper
    """
    input_data = input_data.astype(dtype)
    input_bounds = (input_bounds[0].astype(int), input_bounds[1].astype(int))
    start_block_index = input_bounds[0] // dataset.block_shape
    stop_block_index = (
        input_bounds[1] + dataset.block_shape - 1
    ) // dataset.block_shape
    for i in range(start_block_index[0], stop_block_index[0]):
        for j in range(start_block_index[1], stop_block_index[1]):
            for k in range(start_block_index[2], stop_block_index[2]):
                block_index = np.array([i, j, k], dtype=int)
                block_bounds = (
                    block_index * dataset.block_shape,
                    (block_index + 1) * dataset.block_shape,
                )

                if all(block_bounds[0] >= input_bounds[0]) and all(
                    block_bounds[1] <= input_bounds[1]
                ):
                    # Overwrite block data entirely
                    block_data = input_data[
                        tuple(
                            map(
                                slice,
                                block_bounds[0] - input_bounds[0],
                                block_bounds[1] - input_bounds[0],
                            )
                        )
                    ]
                else:
                    block_data = read(dataset, block_bounds, dtype)

                    intersection_bounds = (
                        np.maximum(block_bounds[0], input_bounds[0]),
                        np.minimum(block_bounds[1], input_bounds[1]),
                    )
                    relative_block_bounds = tuple(
                        map(
                            slice,
                            intersection_bounds[0] - block_bounds[0],
                            intersection_bounds[1] - block_bounds[0],
                        )
                    )
                    relative_data_bounds = tuple(
                        map(
                            slice,
                            intersection_bounds[0] - input_bounds[0],
                            intersection_bounds[1] - input_bounds[0],
                        )
                    )
                    block_data[relative_block_bounds] = input_data[relative_data_bounds]

                dataset.write_block(
                    block_index, block_data.transpose([2, 1, 0]).flatten()
                )
