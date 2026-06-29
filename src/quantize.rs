//! Affine int8 quantisation.
//!
//! The other dtype a TPU loves is int8: an int8 matmul runs at roughly four times
//! the throughput of bf16. To use it you quantise an f32 tensor to int8 with an
//! affine map `q = round(x / scale) + zero_point`, run the integer matmul, then
//! dequantise. This module provides the per-tensor affine map and the lattice
//! conversions, preserving the hardware tiling throughout.

use crate::error::Result;
use crate::lattice::PaddedTileLattice;

/// Parameters of an affine int8 quantisation: `real = scale * (q - zero_point)`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct QuantParams {
    /// The size of one quantisation step in real units.
    pub scale: f32,
    /// The int8 value that maps to real zero.
    pub zero_point: i8,
}

impl QuantParams {
    /// Derive symmetric parameters (zero_point = 0) from a value's magnitude bound.
    ///
    /// `abs_max` is the largest absolute value the tensor takes. The int8 range is
    /// treated as `-127..=127` so that negation stays representable.
    pub fn symmetric(abs_max: f32) -> Self {
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
        QuantParams {
            scale,
            zero_point: 0,
        }
    }

