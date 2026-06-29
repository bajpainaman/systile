# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Holographic Tensor Store** (`holo` module): `HoloMemory`, a key→value
  associative store that holds all entries in superposition inside a single
  fixed-width vector, with batched lookup realized as one systolic matmul.
- `hyper` module: `Hyper`, the bipolar MAP/VSA algebra (bind, bundle, permute,
  similarity) with deterministic atom generation.
- `codebook` module: `Codebook`, a tile-aligned symbol vocabulary whose cleanup is
  a matmul through the systolic engine.
- `holo_kv` and `holo_capacity` examples; `hyper`, `codebook`, and `holo` test
  suites; `HOLOGRAPHIC.md` design note with capacity math and citations.

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

[Unreleased]: https://github.com/bajpainaman/systile/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bajpainaman/systile/releases/tag/v0.1.0
