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

impl<T: Numeric> PaddedTileLattice<T> {
    /// Multiply two lattices, returning the product and dataflow statistics.
    ///
    /// `self` is `m x k`, `rhs` is `k x n`, the result is `m x n`. The product is
    /// accumulated entirely in f32 (like the hardware accumulator) and narrowed to
    /// `T` exactly once per output element.
    pub fn matmul_with_stats(
        &self,
        rhs: &PaddedTileLattice<T>,
    ) -> Result<(PaddedTileLattice<T>, SystolicStats)> {
        if self.cols() != rhs.rows() {
            return Err(LatticeError::ContractionMismatch {
                lhs_cols: self.cols(),
                rhs_rows: rhs.rows(),
            });
        }
        if self.geometry() != rhs.geometry() {
            return Err(LatticeError::GeometryMismatch);
        }

        let geom: Geometry = *self.geometry();
        let m = self.rows();
        let k = self.cols();
        let n = rhs.cols();
        let mxu = geom.mxu;

        // Dense f32 accumulator, exactly like a TPU's matrix accumulator banks.
        let mut acc = vec![0.0f32; m * n];
        let mut stats = SystolicStats::default();

        for i0 in (0..m).step_by(mxu) {
            for j0 in (0..n).step_by(mxu) {
                stats.output_blocks += 1;
                for k0 in (0..k).step_by(mxu) {
                    stats.weight_block_loads += 1;
                    let i_end = (i0 + mxu).min(m);
                    let j_end = (j0 + mxu).min(n);
                    let k_end = (k0 + mxu).min(k);
                    for i in i0..i_end {
                        for j in j0..j_end {
                            let mut sum = acc[i * n + j];
                            for kk in k0..k_end {
                                let a = self.get(i, kk).unwrap().to_acc();
                                let b = rhs.get(kk, j).unwrap().to_acc();
                                sum += a * b;
                                stats.macs += 1;
                            }
                            acc[i * n + j] = sum;
                        }
                    }
                }
            }
        }

        // Account for the padding MACs the hardware would also have to spin through:
        // the array always runs whole mxu-blocks, padding included.
        let padded_macs = (Geometry::round_up(m, mxu) as u64)
            * (Geometry::round_up(n, mxu) as u64)
            * (Geometry::round_up(k, mxu) as u64);
        stats.padding_macs = padded_macs.saturating_sub(stats.macs);

        let dense: Vec<T> = acc.into_iter().map(T::from_acc).collect();
        let out = PaddedTileLattice::from_dense(m, n, &dense, geom)?;
        Ok((out, stats))
    }

