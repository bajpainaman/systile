//! Iterators over a lattice in both logical and hardware-tile order.

use crate::lattice::PaddedTileLattice;

impl<T> PaddedTileLattice<T> {
    /// Borrow the contiguous storage slice of one tile by tile coordinate.
    pub fn tile_slice(&self, tile_row: usize, tile_col: usize) -> &[T] {
        let base = self.layout().tile_base(tile_row, tile_col);
        let len = self.geometry().tile_len();
        &self.as_storage_slice()[base..base + len]
    }

