//! A CPU reference simulator of weight-stationary systolic matmul.
//!
//! A TPU multiplies matrices by streaming one operand through a square grid of
//! multiply-accumulate cells while the other operand sits stationary in the grid.
//! This module reproduces the *blocking and accumulation order* of that dataflow
//! — weights loaded one `mxu x mxu` block at a time, products accumulated into an
//! f32 accumulator — so a result computed here matches what the hardware returns
//! bit-for-bit in the f32 case and closely in the bf16 case. It also reports the
//! work performed so you can reason about utilisation before deploying.

use crate::bf16::Bf16;
use crate::error::{LatticeError, Result};
use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;

/// A numeric element the systolic simulator can stream and accumulate.
pub trait Numeric: Copy + Default {
    /// Widen to the f32 accumulator domain.
    fn to_acc(self) -> f32;
    /// Narrow an f32 accumulator back to the element type.
    fn from_acc(value: f32) -> Self;
}

impl Numeric for f32 {
    #[inline]
    fn to_acc(self) -> f32 {
        self
    }
    #[inline]
    fn from_acc(value: f32) -> Self {
        value
    }
}

impl Numeric for Bf16 {
    #[inline]
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
    #[inline]
    fn from_acc(value: f32) -> Self {
        Bf16::from_f32(value)
    }
}

