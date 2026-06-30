# systile

**Matmul-native data structures & algorithms, written from scratch in Rust.**

[![crates.io](https://img.shields.io/crates/v/systile.svg)](https://crates.io/crates/systile)
[![docs.rs](https://img.shields.io/docsrs/systile)](https://docs.rs/systile)
[![CI](https://github.com/bajpainaman/systile/actions/workflows/ci.yml/badge.svg)](https://github.com/bajpainaman/systile/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

One idea, taken to its conclusion: **build data structures and algorithms whose
dominant operation is a dense matrix multiply.** On a CPU that is usually a bad
trade — a hash map beats a matmul-based map, a queue beats matrix powers. But on a
systolic accelerator (a TPU's matrix unit, a GPU's tensor cores) dense matmul is the
*cheap* primitive and branch-y pointer chasing is the expensive one, so the trade
flips. `systile` is a library of structures built for that world.

It starts with a substrate — the **Padded Tile Lattice**, a tensor laid out the way
a TPU's memory is actually addressed (`8 × 128` `(sublane, lane)` tiles, padding,
bf16/int8 dtypes) with a CPU reference simulator of the systolic matmul — and then
builds a stack of pillars on top of it. You do **not** need a TPU: everything runs
on the CPU model, honestly framed as *matmul-native* (maps efficiently onto the
MXU), not *TPU-exclusive*.

## The pillars

| # | Pillar | Structures | One-line demo |
| --- | --- | --- | --- |
| 0 | **Tensor substrate** | `PaddedTileLattice`, `Bf16`, `systolic`, `quantize` | a matmul in true device layout |
| 1 | **Data as superposition** (VSA) | `HoloMemory`, `HoloSet`, `HoloSequence`, `Resonator` | 200 KV pairs in one 32 KB vector, 1 matmul, 100% recall |
| 2 | **Algorithms as semiring matrix powers** | `TensorGraph`, `semiring` | shortest paths as `⌈log₂n⌉` GEMMs |
| 3 | **Computation as matmul** | `TensorAutomaton` | decide divisibility by matrix multiply |
| 4 | **Learning as bundling** | `HoloClassifier` | train by addition, classify by one matmul |
| 5 | **Retrieval as matmul** | `TensorIndex` | exact k-NN over a corpus in one GEMM |
| 6 | **Probabilistic membership as matmul** | `TensorBloom` | a Bloom filter whose batch query is one matmul |
| 7 | **Sorting as comparison matmul** | `TensorSort` | ranks = `C·1`, sort = `P·x` |
| 8 | **Scan as triangular matmul** | `TensorScan` | prefix sums as `L·x`, `O(1)` depth |
| 9 | **Pattern search as convolution matmul** | `TensorConv` | locate a motif via one im2col matmul |
| 10 | **Frequency as matmul** | `CountMinSketch` | Count-Min estimates as one matmul per hash row |

Every structure reduces its core operation to a matmul through the same systolic
engine. The honest framing, capacity math, and citations live in
**[HOLOGRAPHIC.md](HOLOGRAPHIC.md)**.

## Why a data structure "for TPUs"?

A TPU is not a flat array machine. Three hardware facts drive its data layout, and
`systile` encodes all three:

| Hardware fact | What it forces | Where `systile` handles it |
| --- | --- | --- |
| Vector memory is addressed as `8 × 128` `(sublane, lane)` tiles | Data must be tiled and padded to tile boundaries | [`Geometry`], [`Layout`], [`Shape`] |
| The matrix unit is a `128 × 128` systolic array | Matmul runs in square `mxu` blocks, padding included | [`systolic`] |
| Native dtypes are `bf16` and `int8`, not `f32` | You quantise/narrow before compute, accumulate in `f32` | [`bf16`], [`quantize`] |

Because padding is mandatory, the structure tracks both the *logical* shape you
asked for and the *padded* shape it actually stores, plus a validity [`Mask`] so
reductions and dense round-trips never fold in garbage.

## Quick start

```rust
use systile::prelude::*;

let a = PaddedTileLattice::from_dense(
    2, 3, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], Geometry::TPU_V,
).unwrap();
let b = PaddedTileLattice::from_dense(
    3, 2, &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], Geometry::TPU_V,
).unwrap();

// Matmul runs in the same blocked dataflow a systolic array uses.
let (c, stats) = a.matmul_with_stats(&b).unwrap();
assert_eq!(c.to_dense(), vec![58.0, 64.0, 139.0, 154.0]);
println!("array utilisation: {:.1}%", stats.utilisation() * 100.0);
```

## The headline: a data structure whose operations *are* matmuls

On top of the tiling substrate, `systile` ships an invented container — the
**Holographic Tensor Store** ([`HoloMemory`]) — a key→value map that holds **every
entry summed on top of every other** inside a single fixed-width vector, and
recovers a value by algebra plus one matrix multiply.

```rust
use systile::prelude::*;

let mut book = HoloMemory::new(8192, 1000, 0xC0FFEE); // 8192-dim, 1000 value symbols
for name in 0..200 {
    book.insert(name, (name * 7 + 3) % 1000);          // bind + bundle into ONE vector
}

// Look up all 200 names at once — a single (200 × 8192)·(8192 × 1000) matmul.
let hits = book.batch_get(&(0..200).collect::<Vec<_>>());
let correct = (0..200).filter(|&n| hits[n].0 == (n * 7 + 3) % 1000).count();
assert_eq!(correct, 200); // 100% recall, well under the d/(2 ln M) capacity bound
```

200 entries live in 32 KB of `f32`; lookup of the whole batch is one MXU-shaped
GEMM. On a CPU this is a *worse* map than a hash table — it only pays off where
dense matmul is the cheap primitive and you batch thousands of probes: a TPU. It's
approximate and bounded (`K_max ≈ d / (2 ln M)`), degrading gracefully past
capacity. The full mechanism, capacity math, honest novelty assessment, and
citations are in **[HOLOGRAPHIC.md](HOLOGRAPHIC.md)**. Try it:

```
cargo run --release --example holo_kv          # 200 pairs in one vector, 1 matmul
cargo run --release --example holo_capacity    # recall vs the d/(2 ln M) bound
cargo run --release --example resonator_factor # factor a product with no known factors
cargo run --release --example holo_precision   # f32 vs bf16 cleanup recall
cargo run           --example holo_analogy      # "Dollar of Mexico?" -> peso, zero training
cargo run --release --example graph_paths      # shortest paths as tropical matrix powers
cargo run --release --example automaton_divisibility  # decide divisibility by matmul
cargo run --release --example classifier_demo  # train by bundling, classify by matmul
cargo run --release --example index_search     # exact k-NN search as one matmul
cargo run --release --example bloom_membership # Bloom membership as one matmul
cargo run --release --example sort_by_matmul   # sort via comparison + permutation matmul
cargo run --release --example scan_prefix      # prefix sums as a triangular matmul
cargo run --release --example conv_search      # pattern search as im2col correlation
cargo run --release --example sketch_frequency # Count-Min frequency estimates by matmul
```

## Features

- **A family of matmul-native containers** on a hyperdimensional (VSA) substrate
  (`Hyper` algebra + `Codebook` matmul cleanup):
  - `HoloMemory` — key→value store in superposition; batched lookup is one matmul.
  - `HoloSet` — set membership as a matmul; union by bundling; norm-based cardinality.
  - `HoloSequence` — order via permutation binding; whole-sequence decode in one matmul.
  - `Resonator` — factor a bound product back into its unknown symbols by iterated
    matmul cleanup (an `Mᶠ` search run as a short sequence of GEMMs), with exact
    verification and restarts.
- **`TensorGraph`** — graph algorithms as semiring matrix powers (GraphBLAS-style):
  reachability (boolean), all-pairs shortest paths (tropical/min-plus), and walk
  counting (ordinary) — each in `⌈log₂ n⌉` dense matmuls via repeated squaring.
- **`TensorAutomaton`** — a finite-state machine run as matmuls: one-hot state
  vector × per-symbol transition matrix. Branchless string recognition; a whole
  batch advances with `|alphabet|` masked matmuls per position (e.g. decide
  divisibility by matrix multiply).
- **`HoloClassifier`** — a hyperdimensional classifier: *train by bundling* (no
  gradients, no epochs — fitting is vector addition) and *classify by matmul*
  against the class-prototype matrix. 100% on the synthetic clustering demo.
- **`TensorIndex`** — exact nearest-neighbour / similarity search (the vector-DB
  workload): score a batch of queries against the whole corpus in one
  `(b × dim)·(dim × n)` matmul, then take top-k.
- **`TensorBloom`** — a counting Bloom filter whose batch membership test is one
  matmul of item signatures against the filter's presence vector; no false
  negatives, deletion supported, false-positive rate exposed.
- **`TensorSort`** — sorting as comparison matmul: the rank vector is `C·1` (row
  sums of the pairwise comparison matrix) and the sorted output is `P·x`, an
  `O(n²)`-matmul trade against `O(n log n)` branches.
- **`TensorScan`** — prefix sums as a triangular matmul (`L·x`): inclusive,
  exclusive, and suffix scans with `O(1)` dependency depth.
- **`TensorConv`** — 1-D pattern search as im2col cross-correlation: gather all
  windows and dot them against the kernel in one matmul, then argmax for the match.
- **`CountMinSketch`** — frequency estimation where each row's query is a matmul of
  a one-hot column selection against that row's counters; never underestimates.
- **`PaddedTileLattice<T>`** — the core 2-D tiled tensor, generic over element type.
- **`bf16`** — a from-scratch bfloat16 with round-to-nearest-even and a full set of
  arithmetic / comparison / conversion impls.
- **Systolic matmul simulator** — weight-stationary, `f32`-accumulated, verified
  bit-for-bit against a naive triple loop, and it reports MAC utilisation.
- **Tile-level sparsity** — find and skip the all-zero tiles a kernel would waste
  cycles on.
- **Affine int8 quantisation** — symmetric and asymmetric calibration that
  preserves the hardware tiling end to end.
- **Transpose & relayout** — re-tile the same logical data under a new geometry.
- **Element-wise maps and reductions** — padding-correct by construction.
- `#![forbid(unsafe_code)]`, no required dependencies.

## Examples

```
cargo run --example quickstart
cargo run --example bf16_roundtrip
cargo run --example quantize_matmul
cargo run --example sparsity_report
cargo run --example padding_inspect
cargo bench
```

## Layout, in one picture

A `3 × 5` logical matrix on `Geometry::TPU_V` (8 sublanes × 128 lanes) pads up to a
single `8 × 128` tile. Element `(row, col)` lives at:

```
offset = tile_index * (sublanes * lanes) + sublane * lanes + lane
```

`tile_index` walks tiles in row-major order; within a tile the order is row-major
over `(sublane, lane)`. That is exactly the order a TPU's vector memory expects, so
`as_storage_slice()` is copy-ready.

## Status

`systile` is young and the API may shift before `1.0`. The simulator is a reference
model, not a cycle-accurate one: it reproduces the **blocking and accumulation
order** of a systolic array (and so its numerics), not its timing.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your
option.

[`HoloMemory`]: https://docs.rs/systile/latest/systile/holo/struct.HoloMemory.html
[`Geometry`]: https://docs.rs/systile/latest/systile/geometry/struct.Geometry.html
[`Layout`]: https://docs.rs/systile/latest/systile/layout/struct.Layout.html
[`Shape`]: https://docs.rs/systile/latest/systile/shape/struct.Shape.html
[`Mask`]: https://docs.rs/systile/latest/systile/mask/struct.Mask.html
[`bf16`]: https://docs.rs/systile/latest/systile/bf16/index.html
[`systolic`]: https://docs.rs/systile/latest/systile/systolic/index.html
[`quantize`]: https://docs.rs/systile/latest/systile/quantize/index.html
[`transpose`]: https://docs.rs/systile/latest/systile/transpose/index.html
