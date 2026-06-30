//! Semirings, and matrix multiply over them.
//!
//! A matrix multiply is `C[i,j] = Σ_k A[i,k] · B[k,j]` — but "Σ" and "·" need not
//! be ordinary plus and times. Swap them for a different
//! [semiring](https://en.wikipedia.org/wiki/Semiring) and the *same* dense matmul
//! computes something else entirely:
//!
//! - **Boolean** (`∨`, `∧`): powers of an adjacency matrix give reachability.
//! - **Tropical / min-plus** (`min`, `+`): powers give all-pairs shortest paths.
//! - **Counting** (`+`, `×`): powers count walks.
//!
//! This is the GraphBLAS insight — graph algorithms *are* linear algebra — and it
//! is what lets [`crate::graph::TensorGraph`] run BFS and shortest paths as a short
//! sequence of dense matrix products, the shape an MXU wants.

use crate::error::{LatticeError, Result};
use crate::lattice::PaddedTileLattice;

/// A semiring over `f32`: an additive monoid (`⊕`, `zero`) and a multiplicative
/// monoid (`⊗`, `one`), with `⊗` distributing over `⊕`.
pub trait Semiring {
    /// The additive identity `0̄` (also the annihilator for `⊗`).
    fn zero() -> f32;
    /// The multiplicative identity `1̄`.
    fn one() -> f32;
    /// The additive operation `⊕`.
    fn add(a: f32, b: f32) -> f32;
    /// The multiplicative operation `⊗`.
    fn mul(a: f32, b: f32) -> f32;
}

/// Boolean semiring: `⊕ = OR`, `⊗ = AND`. Non-zero means `true`.
pub struct Boolean;

impl Semiring for Boolean {
    #[inline]
    fn zero() -> f32 {
        0.0
    }
    #[inline]
    fn one() -> f32 {
        1.0
    }
    #[inline]
    fn add(a: f32, b: f32) -> f32 {
        if a != 0.0 || b != 0.0 {
            1.0
        } else {
            0.0
        }
    }
    #[inline]
    fn mul(a: f32, b: f32) -> f32 {
        if a != 0.0 && b != 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

/// Tropical (min-plus) semiring: `⊕ = min`, `⊗ = +`, `zero = +∞`, `one = 0`.
pub struct Tropical;

impl Semiring for Tropical {
    #[inline]
    fn zero() -> f32 {
        f32::INFINITY
    }
    #[inline]
    fn one() -> f32 {
        0.0
    }
    #[inline]
    fn add(a: f32, b: f32) -> f32 {
        if a < b {
            a
        } else {
            b
        }
    }
    #[inline]
    fn mul(a: f32, b: f32) -> f32 {
        a + b
    }
}

/// Counting semiring: ordinary `⊕ = +`, `⊗ = ×`. Powers count walks.
pub struct Counting;

impl Semiring for Counting {
    #[inline]
    fn zero() -> f32 {
        0.0
    }
    #[inline]
    fn one() -> f32 {
        1.0
    }
    #[inline]
    fn add(a: f32, b: f32) -> f32 {
        a + b
    }
    #[inline]
    fn mul(a: f32, b: f32) -> f32 {
        a * b
    }
}

/// Multiply two lattices over the semiring `S`. Shapes must contract
/// (`a.cols() == b.rows()`) and the geometries must match.
pub fn semiring_matmul<S: Semiring>(
    a: &PaddedTileLattice<f32>,
    b: &PaddedTileLattice<f32>,
) -> Result<PaddedTileLattice<f32>> {
    if a.cols() != b.rows() {
        return Err(LatticeError::ContractionMismatch {
            lhs_cols: a.cols(),
            rhs_rows: b.rows(),
        });
    }
    if a.geometry() != b.geometry() {
        return Err(LatticeError::GeometryMismatch);
    }
    let m = a.rows();
    let k = a.cols();
    let n = b.cols();
    let mut out = vec![S::zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = S::zero();
            for kk in 0..k {
                acc = S::add(acc, S::mul(*a.get(i, kk).unwrap(), *b.get(kk, j).unwrap()));
            }
            out[i * n + j] = acc;
        }
    }
    PaddedTileLattice::from_dense(m, n, &out, *a.geometry())
}
