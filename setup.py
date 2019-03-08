#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

setup_requirements = []

test_requirements = []

setup(
    author="William Hunter Patton",
    author_email="pattonw@hhmi.org",
    rust_extensions=[RustExtension("pyn5.pyn5", "Cargo.toml", binding=Binding.PyO3)],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Rust",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Python wrapper around rust-n5.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="pyn5",
    name="pyn5",
    packages=find_packages(include=["pyn5"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/pattonw/pyn5",
    version="0.1.0",
    zip_safe=False,
)
