//! Error type for fallible lattice operations.

use core::fmt;

/// Errors produced when constructing or transforming a [`crate::PaddedTileLattice`].
#[derive(Clone, PartialEq, Eq)]
pub enum LatticeError {
    /// A dense buffer did not contain `rows * cols` elements.
    BufferLengthMismatch {
        /// Number of elements that were expected.
        expected: usize,
        /// Number of elements that were actually supplied.
        actual: usize,
    },
    /// A tile dimension was zero, which is never legal.
    ZeroTileDimension,
    /// Two lattices that had to agree on geometry did not.
    GeometryMismatch,
    /// A matmul was attempted on shapes whose contraction dimensions disagree.
    ContractionMismatch {
        /// Columns of the left-hand operand.
        lhs_cols: usize,
        /// Rows of the right-hand operand.
        rhs_rows: usize,
    },
    /// An index was outside the logical shape of the lattice.
    IndexOutOfBounds {
        /// The row that was requested.
        row: usize,
        /// The column that was requested.
        col: usize,
    },
}

