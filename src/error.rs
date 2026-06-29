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

impl fmt::Debug for LatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LatticeError::BufferLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "BufferLengthMismatch {{ expected: {expected}, actual: {actual} }}"
                )
            }
            LatticeError::ZeroTileDimension => write!(f, "ZeroTileDimension"),
            LatticeError::GeometryMismatch => write!(f, "GeometryMismatch"),
            LatticeError::ContractionMismatch { lhs_cols, rhs_rows } => {
                write!(
                    f,
                    "ContractionMismatch {{ lhs_cols: {lhs_cols}, rhs_rows: {rhs_rows} }}"
                )
            }
            LatticeError::IndexOutOfBounds { row, col } => {
                write!(f, "IndexOutOfBounds {{ row: {row}, col: {col} }}")
            }
        }
    }
}

impl fmt::Display for LatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LatticeError::BufferLengthMismatch { expected, actual } => write!(
                f,
                "dense buffer length mismatch: expected {expected} elements, got {actual}"
            ),
            LatticeError::ZeroTileDimension => {
                write!(f, "tile dimensions must be non-zero")
            }
            LatticeError::GeometryMismatch => {
                write!(f, "operands do not share the same tile geometry")
            }
            LatticeError::ContractionMismatch { lhs_cols, rhs_rows } => write!(
                f,
                "contraction mismatch: lhs has {lhs_cols} columns but rhs has {rhs_rows} rows"
            ),
            LatticeError::IndexOutOfBounds { row, col } => {
                write!(f, "index ({row}, {col}) is out of bounds")
            }
        }
    }
}

