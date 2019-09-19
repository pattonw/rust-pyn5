=====
Usage
=====

To use pyn5 in a project::

    import pyn5

pyn5 exposes an API largely compatible with h5py_.
There are additionally some enums defined to optionally help manage open modes and compression types


.. code-block:: python

    import numpy as np

    from pyn5 import File, Mode, CompressionType

    f = File("path/to/test.n5", mode=Mode.READ_WRITE_CREATE)  # default mode 'a'

    g1 = f.create_group("group1")
    g2 = f.require_group("/group1/group2")
    ds1 = g2.create_dataset(
        "dataset1",
        data=np.random.random((10, 10)),
        chunks=(5, 5),
        compression=CompressionType.GZIP,
        compression_opts=-1
    )  # default compression

    # indexing supports slices, integers, ellipses, and newaxes
    arr = ds1[:, 5, np.newaxis]

    ds1.attrs["key"] = "value"


Differences from h5py
---------------------

* The HDF5_ format is different to N5_; refer to their specifications
* No files are held open, so there is no need to use a context manager (``with``) to open a ``File``

  - But you can use one if you want, for compatibility

* Attributes must be JSON-serializable

  - The default encoder will convert numpy arrays to nested lists; they will remain lists when read

* Compression types are as described in the N5_ spec
* Group and Dataset copying and linking are not supported
* Non-zero fill values, dataset resizing, and named dimensions are not supported


.. _HDF5: https://support.hdfgroup.org/HDF5/doc/H5.format.html
.. _h5py: http://docs.h5py.org/en/stable/
.. _N5: https://github.com/saalfeldlab/n5/#file-system-specification-version-203-snapshot
