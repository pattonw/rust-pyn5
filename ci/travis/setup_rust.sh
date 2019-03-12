#!/bin/sh

set -ex

### Setup Rust toolchain #######################################################

curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain=nightly
export PATH=$PATH:$HOME/.cargo/bin