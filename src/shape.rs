//! Logical and padded shapes.
//!
//! Every lattice tracks two shapes at once: the *logical* shape the user cares
//! about, and the *padded* shape the hardware actually stores. Keeping both lets
//! the lattice mask away the padding when it converts back to a dense view.

use crate::geometry::Geometry;

/// A two-dimensional shape together with the padding implied by a [`Geometry`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// The number of rows the user asked for.
    pub rows: usize,
    /// The number of columns the user asked for.
    pub cols: usize,
    /// The row count after padding up to a sublane-tile boundary.
    pub padded_rows: usize,
    /// The column count after padding up to a lane-tile boundary.
    pub padded_cols: usize,
}

impl Shape {
    /// Derive a padded shape from a logical `rows x cols` and a geometry.
    pub fn new(rows: usize, cols: usize, geom: &Geometry) -> Self {
        Shape {
            rows,
            cols,
            padded_rows: geom.pad_rows(rows),
            padded_cols: geom.pad_cols(cols),
        }
    }

    /// Number of logical elements (`rows * cols`).
    #[inline]
    pub const fn logical_len(&self) -> usize {
        self.rows * self.cols
    }

