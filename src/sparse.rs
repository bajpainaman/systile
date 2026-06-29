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

impl<T: IsZero> PaddedTileLattice<T> {
    /// True if every element of tile `(tile_row, tile_col)` is zero.
    pub fn is_tile_zero(&self, tile_row: usize, tile_col: usize) -> bool {
        self.tile_slice(tile_row, tile_col)
            .iter()
            .all(|x| x.is_zero())
    }

    /// Count how many storage tiles are entirely zero.
    pub fn count_zero_tiles(&self) -> usize {
        self.iter_tiles()
            .filter(|(_, _, slice)| slice.iter().all(|x| x.is_zero()))
            .count()
    }

    /// Collect the `(tile_row, tile_col)` coordinates of every non-zero tile.
    ///
    /// These are exactly the tiles a sparsity-aware kernel needs to feed through
    /// the systolic array; the rest can be skipped.
    pub fn nonzero_tile_coords(&self) -> Vec<(usize, usize)> {
        self.iter_tiles()
            .filter(|(_, _, slice)| !slice.iter().all(|x| x.is_zero()))
            .map(|(r, c, _)| (r, c))
            .collect()
    }

