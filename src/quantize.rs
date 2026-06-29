//! Affine int8 quantisation.
//!
//! The other dtype a TPU loves is int8: an int8 matmul runs at roughly four times
//! the throughput of bf16. To use it you quantise an f32 tensor to int8 with an
//! affine map `q = round(x / scale) + zero_point`, run the integer matmul, then
//! dequantise. This module provides the per-tensor affine map and the lattice
//! conversions, preserving the hardware tiling throughout.

use crate::error::Result;
use crate::lattice::PaddedTileLattice;

