# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0]

First stable release. The public API is now covered by semantic versioning —
breaking changes wait for a `2.0`.

### Summary

`systile` is a library of **17 matmul-native data structures and algorithms** built
on a TPU-style tiled-tensor substrate with a systolic reference engine. Each
structure reduces its core operation to a dense matrix multiply: holographic
key/value, set, and sequence memories and a resonator factoriser; semiring graph
algorithms; a finite automaton; a hyperdimensional classifier; exact k-NN; a Bloom
filter; comparison sort and top-k; prefix scan; convolution search; a Count-Min
sketch; tropical edit distance; PageRank; the DFT; Viterbi decoding; and
scaled-dot-product attention.

### Changed

- Crate `description`, `keywords`, and `categories` updated to reflect the full
  library rather than just the tiling substrate.

## [0.10.0]

### Added

- `pillars_bench` benchmark: a dependency-free micro-benchmark of each pillar's
  headline operation (`cargo bench --bench pillars_bench`).
- `capstone_rag` example: composes pillars into a retrieval-augmented classifier
  (`TensorIndex` → `TensorAttention` → decision), all matmuls, 100% on the demo.

### Changed

- The systolic reference matmul now materialises both operands to flat buffers
  once and streams them contiguously (RHS transposed), making it ~6× faster on the
  headline holographic lookup while preserving the dataflow and results.
- Crate-level documentation rewritten as a pillar landing page.

## [0.9.0]

### Added — spectra, decoding, and attention as matmul

- `dft` module: `TensorDFT`, the discrete Fourier transform as a matmul by the
  Fourier matrix — forward, inverse, complex, and magnitude spectra.
- `viterbi` module: `TensorViterbi`, most-likely HMM state decoding by max-plus
  matmul stepping with back-pointer traceback; adds the `MaxPlus` semiring.
- `attention` module: `TensorAttention`, scaled dot-product attention
  `softmax(QKᵀ/√d)·V` as a soft associative memory.
- `semiring::MaxPlus` (max, +) semiring.
- Examples `dft_spectrum`, `viterbi_decode`, `attention_retrieval`.

## [0.8.0]

### Added — selection, edit distance, and ranking as matmul

- `topk` module: `TensorTopK`, top-k selection by a comparison-count matmul
  (`count = C·1`, keep `count < k`), batched, with `kth_largest`.
- `editdist` module: `TensorEditDistance`, Levenshtein distance as a tropical
  (min-plus) shortest path through the alignment grid, relaxed by iterated min-plus
  matmuls. `distance` and `distance_str`.
- `pagerank` module: `TensorPageRank`, PageRank by power iteration — repeated `M·r`
  matmuls against the column-stochastic Google matrix, with dangling-node handling.
- Examples `topk_select`, `edit_distance`, `pagerank_demo`.

## [0.7.0]

### Added — scan, convolution, and frequency as matmul

- `scan` module: `TensorScan`, prefix sums as a triangular matmul (`L·x`) —
  inclusive, exclusive, suffix, and total.
- `conv` module: `TensorConv`, 1-D pattern search as im2col cross-correlation —
  `correlate`, `best_offset`, `find_all`.
- `sketch` module: `CountMinSketch`, frequency estimation where each hash row's
  batch query is a matmul of a one-hot column selection against that row's
  counters, taking the min across rows. Never underestimates.
- Examples `scan_prefix`, `conv_search`, `sketch_frequency`.

## [0.6.0]

### Added — probabilistic membership and sorting as matmul

- `bloom` module: `TensorBloom`, a counting Bloom filter whose batch membership
  test is one matmul of item signatures against the filter's presence vector. No
  false negatives, supports removal, exposes the false-positive rate.
- `sort` module: `TensorSort`, sorting as a comparison matmul — ranks are `C·1`
  (row sums of the pairwise comparison matrix) and the sort is the permutation
  matmul `P·x`. `argsort`, `sort`, `sort_via_matmul`, `permutation_matrix`.
- Examples `bloom_membership` and `sort_by_matmul`.

## [0.5.0]

### Added — learning and retrieval as matmul

- `classifier` module: `HoloClassifier`, a hyperdimensional classifier that trains
  by bundling samples into per-class prototype vectors (no gradients) and classifies
  a whole batch with one matmul against the prototype matrix. Includes scalar/level
  encoding for quantised features.
- `index` module: `TensorIndex`, an exact nearest-neighbour index that scores a
  batch of queries against the entire corpus in a single matmul and returns top-k
  hits (`Hit`).
- Examples `classifier_demo` (100% on a synthetic clustering task) and
  `index_search` (exact k-NN over 2000 vectors in one GEMM).
- README rewritten around the five-pillar framing.

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

[Unreleased]: https://github.com/bajpainaman/systile/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/bajpainaman/systile/compare/v0.10.0...v1.0.0
[0.10.0]: https://github.com/bajpainaman/systile/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/bajpainaman/systile/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/bajpainaman/systile/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/bajpainaman/systile/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/bajpainaman/systile/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/bajpainaman/systile/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/bajpainaman/systile/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bajpainaman/systile/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bajpainaman/systile/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bajpainaman/systile/releases/tag/v0.1.0
