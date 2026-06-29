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

    #[inline]
    fn bit_index(&self, row: usize, col: usize) -> usize {
        row * self.padded_cols + col
    }

    /// True if `(row, col)` is a logical (non-padding) element.
    #[inline]
    pub fn is_valid(&self, row: usize, col: usize) -> bool {
        if row >= self.padded_rows || col >= self.padded_cols {
            return false;
        }
        let idx = self.bit_index(row, col);
        (self.words[idx / 64] >> (idx % 64)) & 1 == 1
    }

    /// Total number of logical elements covered by this mask.
    #[inline]
    pub fn count_valid(&self) -> usize {
        self.rows * self.cols
    }

    /// Total number of padding elements covered by this mask.
    #[inline]
    pub fn count_padding(&self) -> usize {
        self.padded_rows * self.padded_cols - self.rows * self.cols
    }

    /// True if every padded slot is logical (no padding at all).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count_padding() == 0
    }

    /// Count how many of a tile's slots are logical, for a `sublanes x lanes` tile
    /// whose top-left padded coordinate is `(base_row, base_col)`.
    pub fn count_valid_in_tile(
        &self,
        base_row: usize,
        base_col: usize,
        sublanes: usize,
        lanes: usize,
    ) -> usize {
        let mut count = 0;
        for r in base_row..(base_row + sublanes) {
            for c in base_col..(base_col + lanes) {
                if self.is_valid(r, c) {
                    count += 1;
                }
            }
        }
        count
    }
}

