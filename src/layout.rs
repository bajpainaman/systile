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

