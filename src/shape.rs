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

    /// Number of stored elements after padding (`padded_rows * padded_cols`).
    #[inline]
    pub const fn padded_len(&self) -> usize {
        self.padded_rows * self.padded_cols
    }

    /// Number of padding elements that exist only to fill out edge tiles.
    #[inline]
    pub const fn padding_len(&self) -> usize {
        self.padded_len() - self.logical_len()
    }

    /// Fraction of stored elements that are pure padding, in `0.0..=1.0`.
    #[inline]
    pub fn padding_ratio(&self) -> f64 {
        if self.padded_len() == 0 {
            0.0
        } else {
            self.padding_len() as f64 / self.padded_len() as f64
        }
    }

