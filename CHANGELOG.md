# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0]

### Added — finite-state computation as matmul

- `automaton` module: `TensorAutomaton`, a deterministic finite automaton executed
  as matrix multiplies. The state is a one-hot vector, each symbol is a `Q × Q`
  transition matrix, and a step is `state · T_symbol`. Includes `mod_k` (accept
  binary strings divisible by `k`), single-string `run`/`accepts`, and a branchless
  batched `run_batch`/`batch_accepts` that advances a whole batch with `|alphabet|`
  masked matmuls per position.
- Example `automaton_divisibility` (divisibility decided purely by matmul).

## [0.3.0]

### Added — graph algorithms as semiring matmul

- `semiring` module: the `Semiring` trait with `Boolean`, `Tropical` (min-plus),
  and `Counting` instances, plus `semiring_matmul` over a `PaddedTileLattice`.
- `graph` module: `TensorGraph`, a directed weighted graph whose algorithms are
  matrix powers — reachability/transitive closure (boolean), all-pairs shortest
  paths (tropical), and walk counting (ordinary) — each computed in `⌈log₂ n⌉`
  dense matmuls by repeated squaring.
- `holo::HoloMemory::probe` exposed publicly; `Codebook::cleanup_batch_bf16` and
  `Codebook::superpose` added in support of the holographic and resonator paths.
- Examples `graph_paths`, `holo_precision`, `holo_analogy`.

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

[Unreleased]: https://github.com/bajpainaman/systile/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/bajpainaman/systile/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bajpainaman/systile/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bajpainaman/systile/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bajpainaman/systile/releases/tag/v0.1.0
