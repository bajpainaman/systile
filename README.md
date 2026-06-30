# systile

**A TPU-native tiled tensor data structure, written from scratch in Rust.**

[![crates.io](https://img.shields.io/crates/v/systile.svg)](https://crates.io/crates/systile)
[![docs.rs](https://img.shields.io/docsrs/systile)](https://docs.rs/systile)
[![CI](https://github.com/bajpainaman/systile/actions/workflows/ci.yml/badge.svg)](https://github.com/bajpainaman/systile/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Most tensor libraries store a flat row-major buffer and bolt on a layout pass when
it is time to talk to an accelerator. `systile` inverts that: its core data
structure, the **Padded Tile Lattice**, is laid out the way a Tensor Processing
Unit's memory is addressed from the very first allocation. The flat buffer it owns
is *already* in device order, so handing data to hardware is a `memcpy` rather than
a transpose.

This is a host-side data structure and a CPU reference simulator. You do **not**
need a TPU to use it — the point is to model the constraints a TPU imposes
(mandatory tiling, padding, `(sublane, lane)` addressing, bf16/int8 dtypes, square
matrix-unit blocking) so you can prepare, validate, and reason about layouts before
anything touches real silicon.

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
