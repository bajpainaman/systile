//! Reductions over the logical region.
//!
//! Reductions are where padding bites: a naive sum over the padded buffer would
//! fold in garbage. Every reduction here walks only logical elements, which is the
//! whole reason the lattice keeps a [`crate::mask::Mask`] around.

