//! Element-wise maps and binary combinators over the logical region.
//!
//! These operate only on logical elements; padding is regenerated as
//! `T::default()` so it can never leak into a result. On hardware these map to
//! the vector unit, which is why they preserve tiling exactly.

use crate::error::{LatticeError, Result};
use crate::lattice::PaddedTileLattice;

impl<T: Clone + Default> PaddedTileLattice<T> {
    /// Apply `f` to every logical element, returning a new lattice of the same shape.
    pub fn map<U, F>(&self, mut f: F) -> PaddedTileLattice<U>
    where
        U: Clone + Default,
        F: FnMut(&T) -> U,
    {
        let mut out = PaddedTileLattice::<U>::zeroed(self.rows(), self.cols(), *self.geometry())
            .expect("shape is preserved, so geometry stays valid");
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                let mapped = f(self.get(row, col).unwrap());
                out.set(row, col, mapped).unwrap();
            }
        }
        out
    }

    /// Apply `f` to every logical element in place.
    pub fn map_in_place<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> T,
    {
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                let mapped = f(self.get(row, col).unwrap());
                self.set(row, col, mapped).unwrap();
            }
        }
    }

