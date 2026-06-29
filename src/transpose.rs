//! Transpose and relayout.
//!
//! On real hardware a transpose is a sequence of lane rotations through the
//! cross-lane unit; here we produce the logically-transposed lattice directly.
//! Relayout re-tiles the same logical data under a different [`Geometry`], which
//! is what you do when handing a tensor from the vector unit to the matrix unit.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;

impl<T: Clone + Default> PaddedTileLattice<T> {
    /// Produce the transpose: a `cols x rows` lattice with the same geometry.
    pub fn transpose(&self) -> PaddedTileLattice<T> {
        let mut out = PaddedTileLattice::zeroed(self.cols(), self.rows(), *self.geometry())
            .expect("transposed geometry is valid because the source geometry is");
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                let value = self.get(row, col).unwrap().clone();
                out.set(col, row, value)
                    .expect("transposed coordinate is in bounds");
            }
        }
        out
    }

    /// Re-tile the same logical data under a new geometry.
    pub fn relayout(&self, geom: Geometry) -> crate::error::Result<PaddedTileLattice<T>> {
        let mut out = PaddedTileLattice::zeroed(self.rows(), self.cols(), geom)?;
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                let value = self.get(row, col).unwrap().clone();
                out.set(row, col, value)?;
            }
        }
        Ok(out)
    }

    /// True if transposing twice returns the original logical data. Always true;
    /// exposed as a cheap invariant hook for the `paranoid` feature and tests.
    pub fn is_transpose_involutive(&self) -> bool
    where
        T: PartialEq,
    {
        self.transpose().transpose().to_dense() == self.to_dense()
    }
}
