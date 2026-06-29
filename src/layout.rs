//! Address arithmetic that maps logical `(row, col)` coordinates to the linear
//! offset where the element actually lives in tiled `(sublane, lane)` order.
//!
//! Storage order is *row-major over tiles*, and within each tile *row-major over
//! `(sublane, lane)`*. This is the layout a TPU's vector memory uses, so writing
//! data in this order means a host-to-device copy is a straight `memcpy`.

use crate::geometry::Geometry;
use crate::shape::Shape;

/// Precomputed strides for translating coordinates into storage offsets.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Layout {
    sublanes: usize,
    lanes: usize,
    tile_len: usize,
    tiles_per_row: usize,
}

impl Layout {
    /// Build a layout for a padded shape under a geometry.
    pub fn new(shape: &Shape, geom: &Geometry) -> Self {
        Layout {
            sublanes: geom.sublanes,
            lanes: geom.lanes,
            tile_len: geom.tile_len(),
            tiles_per_row: shape.padded_cols / geom.lanes,
        }
    }

    /// Map a padded coordinate to its linear storage offset.
    ///
    /// The coordinate must already be inside the padded shape; callers that work
    /// with logical coordinates should ensure `row < padded_rows` and likewise
    /// for `col`.
    #[inline]
    pub fn offset(&self, row: usize, col: usize) -> usize {
        let tile_row = row / self.sublanes;
        let tile_col = col / self.lanes;
        let sub = row % self.sublanes;
        let lane = col % self.lanes;
        let tile_index = tile_row * self.tiles_per_row + tile_col;
        tile_index * self.tile_len + sub * self.lanes + lane
    }

    /// Inverse of [`Layout::offset`]: recover the padded coordinate from an offset.
    #[inline]
    pub fn coord(&self, offset: usize) -> (usize, usize) {
        let tile_index = offset / self.tile_len;
        let within = offset % self.tile_len;
        let tile_row = tile_index / self.tiles_per_row;
        let tile_col = tile_index % self.tiles_per_row;
        let sub = within / self.lanes;
        let lane = within % self.lanes;
        let row = tile_row * self.sublanes + sub;
        let col = tile_col * self.lanes + lane;
        (row, col)
    }

    /// The linear offset of the first element of tile `(tile_row, tile_col)`.
    #[inline]
    pub fn tile_base(&self, tile_row: usize, tile_col: usize) -> usize {
        (tile_row * self.tiles_per_row + tile_col) * self.tile_len
    }

    /// Number of storage tiles along the column axis.
    #[inline]
    pub fn tiles_per_row(&self) -> usize {
        self.tiles_per_row
    }
}
