//! Validity masks.
//!
//! Padding fills out the edge tiles of a lattice with elements that have no
//! logical meaning. A [`Mask`] records, per padded coordinate, whether that slot
//! is a real value or padding. It is the bookkeeping that lets a matmul skip
//! garbage and a dense round-trip drop it.

