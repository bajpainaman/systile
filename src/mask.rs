//! Validity masks.
//!
//! Padding fills out the edge tiles of a lattice with elements that have no
//! logical meaning. A [`Mask`] records, per padded coordinate, whether that slot
//! is a real value or padding. It is the bookkeeping that lets a matmul skip
//! garbage and a dense round-trip drop it.

use crate::shape::Shape;

/// A dense bitset over the padded coordinate grid marking logical elements.
#[derive(Clone, PartialEq, Eq)]
pub struct Mask {
    padded_rows: usize,
    padded_cols: usize,
    rows: usize,
    cols: usize,
    words: Vec<u64>,
}

impl Mask {
    /// Build a mask from a shape: a slot is valid iff it lies in the logical region.
    pub fn from_shape(shape: &Shape) -> Self {
        let total = shape.padded_len();
        let word_count = total.div_ceil(64);
        let mut words = vec![0u64; word_count];
        for row in 0..shape.rows {
            for col in 0..shape.cols {
                let idx = row * shape.padded_cols + col;
                words[idx / 64] |= 1u64 << (idx % 64);
            }
        }
        Mask {
            padded_rows: shape.padded_rows,
            padded_cols: shape.padded_cols,
            rows: shape.rows,
            cols: shape.cols,
            words,
        }
    }

