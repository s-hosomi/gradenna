#!/usr/bin/env bash
# Build the optional native FDTD acceleration kernel (rust/ -> cdylib).
#
# This is a *local* accelerator, not part of the Python package: the build
# is never run by pip/pyproject. Run it by hand (or let gradenna.native
# auto-build on first use) when you want the fused Rust 2D TM time loop.
#
# Requires cargo (https://rustup.rs). If cargo is missing the script exits
# non-zero with a clear message and gradenna falls back to the XLA path.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
crate="$here/../rust"

# Find cargo on PATH or in the default rustup location.
cargo_bin="$(command -v cargo || true)"
if [ -z "$cargo_bin" ] && [ -x "$HOME/.cargo/bin/cargo" ]; then
    cargo_bin="$HOME/.cargo/bin/cargo"
fi
if [ -z "$cargo_bin" ]; then
    echo "error: cargo not found. Install Rust from https://rustup.rs to" >&2
    echo "       build the native kernel; gradenna runs fine without it"  >&2
    echo "       (the XLA path is used automatically)."                    >&2
    exit 1
fi

echo "Building native FDTD kernel with $cargo_bin ..."
"$cargo_bin" build --release --manifest-path "$crate/Cargo.toml"
echo "Built: $crate/target/release/ (libgradenna_kernel.{dylib,so})"
