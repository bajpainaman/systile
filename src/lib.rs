//! # systile — matmul-native data structures & algorithms
//!
//! `systile` takes one idea to its conclusion: **build data structures and
//! algorithms whose dominant operation is a dense matrix multiply.** On a CPU that
//! is often a bad trade, but on a systolic accelerator (a TPU's matrix unit, a
//! GPU's tensor cores) dense matmul is the cheap primitive and branch-y pointer
//! chasing is the expensive one — so the trade flips.
//!
//! It begins with a substrate — the [`PaddedTileLattice`], a tensor laid out the
//! way a TPU's memory is actually addressed (`8 × 128` `(sublane, lane)` tiles,
//! padding, bf16/int8 dtypes) with a CPU reference simulator of the systolic matmul
//! ([`systolic`]) — and builds a stack of pillars on top of it.
//!
//! ## The pillars
//!
//! | Pillar | Type | Idea |
//! | --- | --- | --- |
//! | [`holo`], [`holoset`], [`sequence`], [`resonator`] | data | hold a whole structure in *superposition*, recover by matmul cleanup |
//! | [`graph`], [`semiring`] | algorithm | graph algorithms as semiring matrix powers |
//! | [`automaton`] | computation | a finite-state machine run as matmuls |
//! | [`classifier`] | learning | train by bundling, classify by matmul |
//! | [`index`] | retrieval | exact k-NN as one matmul over the corpus |
//! | [`bloom`] | membership | a Bloom filter whose query is a matmul |
//! | [`sort`], [`topk`] | order | sort and select via comparison matmuls |
//! | [`scan`] | scan | prefix sums as a triangular matmul |
//! | [`conv`] | search | pattern search as im2col cross-correlation |
//! | [`sketch`] | frequency | Count-Min estimation as a matmul per hash row |
//! | [`editdist`] | strings | Levenshtein as a tropical (min-plus) shortest path |
//! | [`pagerank`] | ranking | PageRank as power iteration |
//! | [`dft`] | spectra | the discrete Fourier transform as a Fourier-matrix matmul |
//! | [`viterbi`] | decoding | most-likely HMM path as max-plus matmul stepping |
//! | [`attention`] | retrieval | scaled dot-product attention as a soft memory |
//!
//! Everything is honestly *matmul-native* (maps efficiently onto the MXU), not
//! *TPU-exclusive*: it all runs on the CPU reference model. The full design
//! rationale, capacity math, and citations are in `HOLOGRAPHIC.md`.
//!
//! Designing around the hardware buys, for the substrate:
//!
//! 1. **Zero-copy handoff.** [`PaddedTileLattice::as_storage_slice`] is already in
//!    device order; moving it to a TPU is a `memcpy`, not a transpose.
//! 2. **Honest padding.** The structure tracks logical vs. padded shape and keeps
//!    a [`Mask`], so reductions and dense round-trips never fold in garbage.
//! 3. **Hardware-shaped operations.** Matmul, sparsity, quantisation, and transpose
//!    are all expressed in terms of tiles and `mxu` blocks.
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

pub mod attention;
pub mod automaton;
pub mod bf16;
pub mod bloom;
pub mod classifier;
pub mod codebook;
pub mod conv;
pub mod dft;
pub mod editdist;
pub mod elementwise;
pub mod error;
pub mod geometry;
pub mod graph;
pub mod holo;
pub mod holoset;
pub mod hyper;
pub mod index;
pub mod iter;
pub mod lattice;
pub mod layout;
pub mod mask;
pub mod pagerank;
pub mod prelude;
pub mod quantize;
pub mod reduce;
pub mod resonator;
pub mod scan;
pub mod semiring;
pub mod sequence;
pub mod shape;
pub mod sketch;
pub mod sort;
pub mod sparse;
pub mod systolic;
pub mod topk;
pub mod transpose;
pub mod viterbi;

pub use attention::TensorAttention;
pub use automaton::TensorAutomaton;
pub use bf16::Bf16;
pub use bloom::TensorBloom;
pub use classifier::HoloClassifier;
pub use codebook::Codebook;
pub use conv::TensorConv;
pub use dft::TensorDFT;
pub use editdist::TensorEditDistance;
pub use error::{LatticeError, Result};
pub use geometry::Geometry;
pub use graph::TensorGraph;
pub use holo::HoloMemory;
pub use holoset::HoloSet;
pub use hyper::Hyper;
pub use index::{Hit, TensorIndex};
pub use lattice::PaddedTileLattice;
pub use mask::Mask;
pub use pagerank::TensorPageRank;
pub use quantize::QuantParams;
pub use resonator::{Factorization, Resonator};
pub use scan::TensorScan;
pub use sequence::HoloSequence;
pub use shape::Shape;
pub use sketch::CountMinSketch;
pub use sort::TensorSort;
pub use systolic::SystolicStats;
pub use topk::TensorTopK;
pub use viterbi::TensorViterbi;
