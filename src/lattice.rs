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

/// A tensor laid out for systolic hardware: logically `rows x cols`, physically a
/// padded lattice of tiles.
#[derive(Clone)]
pub struct PaddedTileLattice<T> {
    geom: Geometry,
    shape: Shape,
    layout: Layout,
    mask: Mask,
    data: Vec<T>,
}

impl<T> PaddedTileLattice<T> {
    /// The geometry this lattice was built with.
    #[inline]
    pub fn geometry(&self) -> &Geometry {
        &self.geom
    }

    /// The logical-and-padded shape of this lattice.
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The address layout used to translate coordinates into storage offsets.
    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

