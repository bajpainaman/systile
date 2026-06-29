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

