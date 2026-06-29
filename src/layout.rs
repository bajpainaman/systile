//! Address arithmetic that maps logical `(row, col)` coordinates to the linear
//! offset where the element actually lives in tiled `(sublane, lane)` order.
//!
//! Storage order is *row-major over tiles*, and within each tile *row-major over
//! `(sublane, lane)`*. This is the layout a TPU's vector memory uses, so writing
//! data in this order means a host-to-device copy is a straight `memcpy`.

use crate::geometry::Geometry;
use crate::shape::Shape;

