//! Tile geometry: the hardware-dictated tile shape every lattice is built around.
//!
//! A TPU does not see a flat array. Its vector memory is addressed as a grid of
//! `(sublane, lane)` slots — classically 8 sublanes by 128 lanes — and its matrix
//! unit consumes square `mxu x mxu` blocks (classically 128x128). A
//! [`Geometry`] captures those three numbers so the rest of the crate can pad,
//! lay out, and iterate in the order the hardware expects.

use crate::error::{LatticeError, Result};

/// The hardware tile shape a lattice is padded and laid out to.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// Rows per storage tile (the sublane count of a vector register).
    pub sublanes: usize,
    /// Columns per storage tile (the lane count of a vector register).
    pub lanes: usize,
    /// Side length of the square systolic matrix-unit block.
    pub mxu: usize,
}

impl Geometry {
    /// The canonical TPU v-class geometry: 8 sublanes, 128 lanes, 128x128 MXU.
    pub const TPU_V: Geometry = Geometry {
        sublanes: 8,
        lanes: 128,
        mxu: 128,
    };

    /// A small geometry handy for tests and documentation examples.
    pub const TINY: Geometry = Geometry {
        sublanes: 2,
        lanes: 4,
        mxu: 4,
    };

    /// Build a geometry, validating that every dimension is non-zero.
    pub fn new(sublanes: usize, lanes: usize, mxu: usize) -> Result<Self> {
        if sublanes == 0 || lanes == 0 || mxu == 0 {
            return Err(LatticeError::ZeroTileDimension);
        }
        Ok(Geometry {
            sublanes,
            lanes,
            mxu,
        })
    }

    /// Number of elements in a single storage tile (`sublanes * lanes`).
    #[inline]
    pub const fn tile_len(&self) -> usize {
        self.sublanes * self.lanes
    }

    /// Round `n` up to the next multiple of `to`.
    #[inline]
    pub const fn round_up(n: usize, to: usize) -> usize {
        n.div_ceil(to) * to
    }

    /// Pad a row count up to a whole number of sublane tiles.
    #[inline]
    pub const fn pad_rows(&self, rows: usize) -> usize {
        Geometry::round_up(rows, self.sublanes)
    }

