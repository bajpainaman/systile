//! The `PaddedTileLattice`: a dense 2-D tensor stored as a padded grid of
//! hardware tiles in `(sublane, lane)` order.
//!
//! This is the central data structure of the crate. It owns one contiguous
//! buffer whose bytes are already in the order a TPU's vector memory wants, plus
//! the [`Shape`], [`Layout`], and [`Mask`] needed to present a clean logical view
//! on top of that hardware layout.

use crate::error::{LatticeError, Result};
use crate::geometry::Geometry;
use crate::layout::Layout;
use crate::mask::Mask;
use crate::shape::Shape;

