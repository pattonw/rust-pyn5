#!/bin/sh

set -ex

### Setup Rust toolchain #######################################################

curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain=$TRAVIS_RUST_VERSION
export PATH=$PATH:$HOME/.cargo/bin
rustup target add x86_64-unknown-linux-musl

set +ex
