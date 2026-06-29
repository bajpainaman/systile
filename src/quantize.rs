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

    /// Derive asymmetric parameters from an observed `[min, max]` range, mapping
    /// it onto the full `-128..=127` int8 interval.
    pub fn asymmetric(min: f32, max: f32) -> Self {
        let span = (max - min).max(f32::MIN_POSITIVE);
        let scale = span / 255.0;
        // Solve for the zero point that sends `min` to -128.
        let zp = (-128.0 - min / scale).round().clamp(-128.0, 127.0);
        QuantParams {
            scale,
            zero_point: zp as i8,
        }
    }

    /// Quantise a single real value to int8 with saturation.
    #[inline]
    pub fn quantize(&self, value: f32) -> i8 {
        let q = (value / self.scale).round() + self.zero_point as f32;
        q.clamp(-128.0, 127.0) as i8
    }

    /// Dequantise a single int8 value back to real units.
    #[inline]
    pub fn dequantize(&self, q: i8) -> f32 {
        self.scale * (q as f32 - self.zero_point as f32)
    }
}

impl PaddedTileLattice<f32> {
    /// Find the largest absolute logical value, useful for symmetric calibration.
    pub fn abs_max(&self) -> f32 {
        let mut m = 0.0f32;
        for (_, _, v) in self.iter_logical() {
            m = m.max(v.abs());
        }
        m
    }

    /// Quantise this f32 lattice to an int8 lattice with the given parameters,
    /// preserving geometry and padding layout.
    pub fn quantize(&self, params: QuantParams) -> Result<PaddedTileLattice<i8>> {
        let mut out = PaddedTileLattice::<i8>::zeroed(self.rows(), self.cols(), *self.geometry())?;
        for (row, col, v) in self.iter_logical() {
            out.set(row, col, params.quantize(v))?;
        }
        Ok(out)
    }
}

