====
pyn5
====


.. image:: https://img.shields.io/pypi/v/pyn5.svg
        :target: https://pypi.python.org/pypi/pyn5

.. image:: https://img.shields.io/pypi/pyversions/pyn5.svg
        :target: https://pypi.python.org/pypi/pyn5

.. image:: https://travis-ci.org/pattonw/rust-pyn5.svg?branch=master
        :target: https://travis-ci.org/pattonw/rust-pyn5

.. image:: https://readthedocs.org/projects/pyn5/badge/?version=latest
        :target: https://rust-pyn5.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Python wrapper around rust-n5.


* Free software: MIT license
* Documentation: https://rust-pyn5.readthedocs.io.

Installation
------------

``pip install pyn5`` installs pre-compiled wheels.
To build from source, you need

* `maturin`_
* rust_ compiler nightly-2019-07-19 (some more recent nightly compilers may also work)

Features
--------

* h5py_ -like interface

Related projects
----------------

* N5_ (file system format spec and reference implementation in java)
* `rust-n5`_ (implementation in rust, used in pyn5)
* zarr_ (similar chunked array storage library and format, supports some N5 features)
* z5_ (C++ implementation of zarr and N5 with python bindings, depends on conda)
* h5py_ (hierarchical array storage)
* `h5py_like`_ (ABCs for APIs like h5py)

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _N5: https://github.com/saalfeldlab/n5/
.. _rust-n5: https://github.com/aschampion/rust-n5/
.. _zarr: https://zarr-developers.github.io/
.. _z5: https://github.com/constantinpape/z5/
.. _maturin: https://pypi.org/project/maturin/
.. _rust: https://www.rust-lang.org/tools/install
.. _h5py: https://www.h5py.org/
.. _h5py_like: https://github.com/clbarnes/h5py_like
