//! Reductions over the logical region.
//!
//! Reductions are where padding bites: a naive sum over the padded buffer would
//! fold in garbage. Every reduction here walks only logical elements, which is the
//! whole reason the lattice keeps a [`crate::mask::Mask`] around.

use crate::lattice::PaddedTileLattice;

impl PaddedTileLattice<f32> {
    /// Sum of every logical element.
    pub fn sum(&self) -> f32 {
        self.iter_logical().map(|(_, _, v)| v).sum()
    }

    /// Largest logical element, or `None` if the lattice is empty.
    pub fn max(&self) -> Option<f32> {
        self.iter_logical()
            .map(|(_, _, v)| v)
            .fold(None, |acc, v| match acc {
                None => Some(v),
                Some(m) => Some(if v > m { v } else { m }),
            })
    }

