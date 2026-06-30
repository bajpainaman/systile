//! `TensorScan` — prefix sums as a triangular matmul.
//!
//! A cumulative sum is usually a sequential loop with a carried accumulator. As a
//! matmul it is a single product with a triangular matrix of ones:
//!
//! ```text
//! inclusive prefix:  y = L · x      L[i,j] = 1 if j ≤ i
//! exclusive prefix:  y = Lˢ · x     Lˢ[i,j] = 1 if j < i
//! suffix sum:        y = U · x      U[i,j] = 1 if j ≥ i
//! ```
//!
//! It is the `O(n²)`-work, `O(1)`-depth form of a scan — exactly the trade that
//! makes sense when the matmul is free and the sequential dependency is the
//! enemy.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;

/// A matmul-based prefix-sum (scan) operator.
#[derive(Clone, Copy, Debug)]
pub struct TensorScan {
    geom: Geometry,
}

impl Default for TensorScan {
    fn default() -> Self {
        TensorScan {
            geom: Geometry::TPU_V,
        }
    }
}

impl TensorScan {
    /// Create a scan operator with the default tile geometry.
    pub fn new() -> Self {
        TensorScan::default()
    }

    /// Multiply a triangular ones-matrix by `x`. `strict` drops the diagonal;
    /// `upper` puts the ones above it (a suffix scan).
    fn triangular_apply(&self, x: &[f32], strict: bool, upper: bool) -> Vec<f32> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let mut dense = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let keep = if upper {
                    if strict {
                        j > i
                    } else {
                        j >= i
                    }
                } else if strict {
                    j < i
                } else {
                    j <= i
                };
                if keep {
                    dense[i * n + j] = 1.0;
                }
            }
        }
        let tri = PaddedTileLattice::from_dense(n, n, &dense, self.geom).unwrap();
        let xv = PaddedTileLattice::from_dense(n, 1, x, self.geom).unwrap();
        tri.matmul(&xv).unwrap().to_dense()
    }

    /// Inclusive prefix sum: `y[i] = Σ_{j ≤ i} x[j]`.
    pub fn inclusive(&self, x: &[f32]) -> Vec<f32> {
        self.triangular_apply(x, false, false)
    }

    /// Exclusive prefix sum: `y[i] = Σ_{j < i} x[j]` (`y[0] = 0`).
    pub fn exclusive(&self, x: &[f32]) -> Vec<f32> {
        self.triangular_apply(x, true, false)
    }

    /// Suffix sum: `y[i] = Σ_{j ≥ i} x[j]`.
    pub fn suffix(&self, x: &[f32]) -> Vec<f32> {
        self.triangular_apply(x, false, true)
    }

    /// The total sum (the last inclusive prefix), or `0` for an empty input.
    pub fn total(&self, x: &[f32]) -> f32 {
        self.inclusive(x).last().copied().unwrap_or(0.0)
    }
}
