import json
from pathlib import Path
import logging

from .libpyn5 import (
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

