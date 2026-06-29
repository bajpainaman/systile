//! Logical and padded shapes.
//!
//! Every lattice tracks two shapes at once: the *logical* shape the user cares
//! about, and the *padded* shape the hardware actually stores. Keeping both lets
//! the lattice mask away the padding when it converts back to a dense view.

use crate::geometry::Geometry;

