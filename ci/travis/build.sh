#!/bin/sh

export CIBW_BEFORE_BUILD='pip install setuptools-rust && source {project}/ci/travis/setup_rust.sh'
export CIBW_BUILD=$CIBW_BUILD
export CIBW_ENVIRONMENT='PATH="$HOME/.rust/bin:$HOME/.cargo/bin:$PATH"'
cibuildwheel --output-dir wheelhouse

pip install twine
python -m twine upload wheelhouse/*.whl