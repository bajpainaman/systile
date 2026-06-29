//! Tile-level sparsity.
//!
//! Systolic hardware pays the same cost for a tile of zeros as for a tile of real
//! work, so the highest-value sparsity optimisation is to *skip whole tiles* that
//! are entirely zero. This module finds those tiles.

use crate::bf16::Bf16;
use crate::lattice::PaddedTileLattice;

/// A type that knows what its additive-identity "zero" is.
pub trait IsZero {
    /// True if this value is the additive identity.
    fn is_zero(&self) -> bool;
}

impl IsZero for f32 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }
}

impl IsZero for f64 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }
}

impl IsZero for i8 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0
    }
}

impl IsZero for i32 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0
    }
}

impl IsZero for Bf16 {
    #[inline]
    fn is_zero(&self) -> bool {
        Bf16::is_zero(*self)
    }
}

