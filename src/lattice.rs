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

    /// The validity mask separating logical elements from padding.
    #[inline]
    pub fn mask(&self) -> &Mask {
        &self.mask
    }

    /// Logical row count.
    #[inline]
    pub fn rows(&self) -> usize {
        self.shape.rows
    }

    /// Logical column count.
    #[inline]
    pub fn cols(&self) -> usize {
        self.shape.cols
    }

    /// Number of logical elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.shape.logical_len()
    }

    /// True if the lattice has no logical elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.shape.logical_len() == 0
    }

    /// Number of stored elements including padding.
    #[inline]
    pub fn padded_len(&self) -> usize {
        self.shape.padded_len()
    }

    /// Number of storage tiles in the lattice.
    #[inline]
    pub fn num_tiles(&self) -> usize {
        (self.shape.padded_rows / self.geom.sublanes) * (self.shape.padded_cols / self.geom.lanes)
    }

    /// Borrow the raw padded storage buffer in tiled order. This is exactly the
    /// byte sequence you would copy to device memory.
    #[inline]
    pub fn as_storage_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutably borrow the raw padded storage buffer in tiled order.
    #[inline]
    pub fn as_storage_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Borrow the logical element at `(row, col)`, or `None` if out of bounds.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if !self.shape.contains(row, col) {
            return None;
        }
        Some(&self.data[self.layout.offset(row, col)])
    }

    /// Mutably borrow the logical element at `(row, col)`, or `None` if out of bounds.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if !self.shape.contains(row, col) {
            return None;
        }
        let off = self.layout.offset(row, col);
        Some(&mut self.data[off])
    }

    /// Overwrite the logical element at `(row, col)`.
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<()> {
        if !self.shape.contains(row, col) {
            return Err(LatticeError::IndexOutOfBounds { row, col });
        }
        let off = self.layout.offset(row, col);
        self.data[off] = value;
        Ok(())
    }
}

impl<T: Clone + Default> PaddedTileLattice<T> {
    /// Allocate a zero-padded lattice of the given logical shape, with every slot
    /// (logical and padding alike) set to `T::default()`.
    pub fn zeroed(rows: usize, cols: usize, geom: Geometry) -> Result<Self> {
        if geom.sublanes == 0 || geom.lanes == 0 || geom.mxu == 0 {
            return Err(LatticeError::ZeroTileDimension);
        }
        let shape = Shape::new(rows, cols, &geom);
        let layout = Layout::new(&shape, &geom);
        let mask = Mask::from_shape(&shape);
        let data = vec![T::default(); shape.padded_len()];
        Ok(PaddedTileLattice {
            geom,
            shape,
            layout,
            mask,
            data,
        })
    }

    /// Build a lattice from a dense, logical, row-major buffer.
    ///
    /// The buffer must contain exactly `rows * cols` elements. Padding slots are
    /// filled with `T::default()`.
    pub fn from_dense(rows: usize, cols: usize, dense: &[T], geom: Geometry) -> Result<Self> {
        if dense.len() != rows * cols {
            return Err(LatticeError::BufferLengthMismatch {
                expected: rows * cols,
                actual: dense.len(),
            });
        }
        let mut lattice = PaddedTileLattice::zeroed(rows, cols, geom)?;
        for row in 0..rows {
            for col in 0..cols {
                let off = lattice.layout.offset(row, col);
                lattice.data[off] = dense[row * cols + col].clone();
            }
        }
        Ok(lattice)
    }

    /// Materialise a dense, logical, row-major buffer, dropping all padding.
    pub fn to_dense(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.shape.logical_len());
        for row in 0..self.shape.rows {
            for col in 0..self.shape.cols {
                out.push(self.data[self.layout.offset(row, col)].clone());
            }
        }
        out
    }

    /// Overwrite every padding slot with `value`, leaving logical elements intact.
    ///
    /// Useful before a reduction where the padding identity matters (e.g. fill
    /// with the additive identity before a sum, or a large negative before a max).
    pub fn fill_padding(&mut self, value: T) {
        for off in 0..self.data.len() {
            let (row, col) = self.layout.coord(off);
            if !self.shape.contains(row, col) {
                self.data[off] = value.clone();
            }
        }
    }
}

impl<T> core::fmt::Debug for PaddedTileLattice<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PaddedTileLattice")
            .field("shape", &self.shape)
            .field("geometry", &self.geom)
            .field("tiles", &self.num_tiles())
            .finish()
    }
}
