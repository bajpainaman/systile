# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0]

### Added — matmul-native data structures

A family of containers whose dominant operation is a dense matmul on the systolic
engine, built on a new hyperdimensional (VSA) substrate:

- `hyper` module: `Hyper`, the bipolar MAP/VSA algebra (bind, bundle, permute,
  similarity) with deterministic atom generation.
- `codebook` module: `Codebook`, a tile-aligned symbol vocabulary whose cleanup
  (`scores_batch`, `superpose`) is a matmul through the systolic engine.
- **Holographic Tensor Store** (`holo`): `HoloMemory`, a key→value store holding
  all entries in superposition in one vector; batched lookup is one matmul.
- **Holographic Set** (`holoset`): `HoloSet`, set membership as a matmul, with
  union by bundling and a squared-norm cardinality estimate.
- **Holographic Sequence** (`sequence`): `HoloSequence`, order encoded by
  permutation binding; whole-sequence decode in one batched matmul.
- **Resonator Network** (`resonator`): `Resonator`, factoring a bound product back
  into its unknown symbols by iterated matmul cleanup, with exact recomposition
  checking and random restarts.
- Examples `holo_kv`, `holo_capacity`, `resonator_factor`; test suites for every
  new module; `HOLOGRAPHIC.md` design note with capacity math and citations.

## [0.1.0]

### Added

- `PaddedTileLattice<T>`, the core TPU-native tiled tensor data structure.
- `Geometry`, `Shape`, `Layout`, and `Mask` for hardware-shaped tiling and padding.
- A from-scratch `bf16` software float with round-to-nearest-even.
- A weight-stationary systolic matmul simulator with dataflow statistics.
- Tile-level sparsity detection and reporting.
- Affine int8 quantisation (symmetric and asymmetric).
- Transpose, relayout, element-wise maps, and padding-correct reductions.
- Examples, integration tests, and a dependency-free benchmark harness.

[Unreleased]: https://github.com/bajpainaman/systile/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/bajpainaman/systile/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bajpainaman/systile/releases/tag/v0.1.0
