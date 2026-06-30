//! # systile — a TPU-native tiled tensor data structure
//!
//! `systile` provides the [`PaddedTileLattice`], a two-dimensional tensor whose
//! in-memory representation is dictated by the hardware that consumes it: a
//! Tensor Processing Unit. Where a normal matrix type stores a flat row-major
//! buffer, a lattice stores a *padded grid of tiles* in `(sublane, lane)` order,
//! which is exactly the layout a TPU's vector memory addresses and its matrix
//! unit consumes.
//!
//! Designing the data structure around the hardware — rather than bolting a
//! layout pass on afterwards — buys three things:
//!
//! 1. **Zero-copy handoff.** [`PaddedTileLattice::as_storage_slice`] is already in
//!    device order; moving it to a TPU is a `memcpy`, not a transpose.
//! 2. **Honest padding.** The structure tracks logical vs. padded shape and keeps
//!    a [`Mask`], so reductions and dense round-trips never fold in garbage.
//! 3. **Hardware-shaped operations.** Matmul ([`systolic`]), sparsity
//!    ([`sparse`]), quantisation ([`quantize`]), and transpose ([`transpose`]) are
//!    all expressed in terms of tiles and `mxu` blocks.
//!
//! ## Quick start
//!
//! ```
//! use systile::prelude::*;
//!
//! // A 3x5 matrix on the canonical TPU geometry pads up to an 8x128 tile.
//! let a = PaddedTileLattice::from_dense(
//!     2, 3,
//!     &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
//!     Geometry::TPU_V,
//! ).unwrap();
//! let b = PaddedTileLattice::from_dense(
//!     3, 2,
//!     &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0],
//!     Geometry::TPU_V,
//! ).unwrap();
//!
//! let c = a.matmul(&b).unwrap();
//! assert_eq!(c.to_dense(), vec![58.0, 64.0, 139.0, 154.0]);
//! ```
//!
//! See the `examples/` directory for end-to-end walkthroughs.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod automaton;
pub mod bf16;
pub mod codebook;
pub mod elementwise;
pub mod error;
pub mod geometry;
pub mod graph;
pub mod holo;
pub mod holoset;
pub mod hyper;
pub mod iter;
pub mod lattice;
pub mod layout;
pub mod mask;
pub mod prelude;
pub mod quantize;
pub mod reduce;
pub mod resonator;
pub mod semiring;
pub mod sequence;
pub mod shape;
pub mod sparse;
pub mod systolic;
pub mod transpose;

pub use automaton::TensorAutomaton;
pub use bf16::Bf16;
pub use codebook::Codebook;
pub use error::{LatticeError, Result};
pub use geometry::Geometry;
pub use graph::TensorGraph;
pub use holo::HoloMemory;
pub use holoset::HoloSet;
pub use hyper::Hyper;
pub use lattice::PaddedTileLattice;
pub use mask::Mask;
pub use quantize::QuantParams;
pub use resonator::{Factorization, Resonator};
pub use sequence::HoloSequence;
pub use shape::Shape;
pub use systolic::SystolicStats;
