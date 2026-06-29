# systile

**A TPU-native tiled tensor data structure, written from scratch in Rust.**

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

## Features

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

