//! Tile geometry: the hardware-dictated tile shape every lattice is built around.
//!
//! A TPU does not see a flat array. Its vector memory is addressed as a grid of
//! `(sublane, lane)` slots — classically 8 sublanes by 128 lanes — and its matrix
//! unit consumes square `mxu x mxu` blocks (classically 128x128). A
//! [`Geometry`] captures those three numbers so the rest of the crate can pad,
//! lay out, and iterate in the order the hardware expects.

use crate::error::{LatticeError, Result};

