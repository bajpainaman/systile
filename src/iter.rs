//! Iterators over a lattice in both logical and hardware-tile order.

use crate::lattice::PaddedTileLattice;

impl<T> PaddedTileLattice<T> {
    /// Borrow the contiguous storage slice of one tile by tile coordinate.
    pub fn tile_slice(&self, tile_row: usize, tile_col: usize) -> &[T] {
        let base = self.layout().tile_base(tile_row, tile_col);
        let len = self.geometry().tile_len();
        &self.as_storage_slice()[base..base + len]
    }

    /// Number of storage tiles along the row axis.
    #[inline]
    pub fn tile_rows(&self) -> usize {
        self.shape().padded_rows / self.geometry().sublanes
    }

    /// Number of storage tiles along the column axis.
    #[inline]
    pub fn tile_cols(&self) -> usize {
        self.shape().padded_cols / self.geometry().lanes
    }

    /// Iterate every tile in row-major tile order, yielding
    /// `(tile_row, tile_col, slice)`.
    pub fn iter_tiles(&self) -> TileIter<'_, T> {
        TileIter {
            lattice: self,
            next: 0,
            total: self.num_tiles(),
        }
    }
}

