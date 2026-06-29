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

/// Work counters describing a single simulated matmul.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SystolicStats {
    /// Total scalar multiply-accumulate operations performed.
    pub macs: u64,
    /// Number of stationary `mxu x mxu` weight blocks loaded into the array.
    pub weight_block_loads: u64,
    /// Number of `mxu x mxu` output blocks produced.
    pub output_blocks: u64,
    /// MAC slots that were spent on padding rather than logical data.
    pub padding_macs: u64,
}

impl SystolicStats {
    /// Total MAC slots the array spun through, useful plus padding.
    pub fn total_macs(&self) -> u64 {
        self.macs + self.padding_macs
    }

    /// Fraction of MAC work that did useful (non-padding) computation.
    pub fn utilisation(&self) -> f64 {
        let total = self.total_macs();
        if total == 0 {
            0.0
        } else {
            self.macs as f64 / total as f64
        }
    }
}

