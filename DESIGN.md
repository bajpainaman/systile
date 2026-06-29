# Design notes

This document explains *why* `systile` is shaped the way it is. The short version:
a TPU does not see a flat array, so a data structure built for a TPU should not
pretend it does.

## The problem with row-major

The default in-memory representation of a matrix is a single row-major buffer.
That is the right choice for a CPU with deep caches and a prefetcher. It is the
wrong choice for a systolic accelerator, which wants its operands pre-tiled, padded
to fixed boundaries, and addressed by `(sublane, lane)`. When you store row-major
and target a TPU, *every* handoff pays for a layout transform.

`systile` pays that cost once, at construction, and never again.

## Three hardware facts

The whole design follows from three properties of the hardware:

1. **Vector memory is tiled.** It is addressed as a grid of `8 × 128`
   `(sublane, lane)` tiles, not as a flat span.
2. **The matrix unit is square.** Matmul is performed by a `128 × 128` systolic
   array, so contraction happens in `mxu`-sized blocks whether your data fills them
   or not.
3. **The native dtypes are narrow.** `bf16` and `int8` feed the array; `f32` is the
   accumulator, not the input.

`Geometry` captures the first two as `(sublanes, lanes, mxu)`. The dtype facts live
in the `bf16` and `quantize` modules.

## Logical vs. padded shape

Because tiling is mandatory, a `3 × 5` matrix cannot be stored as 15 elements. It
is stored as a full `8 × 128` tile with 1009 padding slots. The structure therefore
tracks two shapes at once — the logical one the user reasons about and the padded
one the hardware stores — in the `Shape` type.

## Why keep a mask

Padding is not free of consequences: a naive `sum` over the padded buffer would add
in 1009 zeros (fine) but a naive `max` over a buffer whose padding was filled with
a sentinel would return the sentinel (not fine). The `Mask` records exactly which
slots are logical, so every reduction and every dense round-trip can ignore padding
deliberately rather than by luck.

## The address map

Element `(row, col)` lives at:

```
tile_index = (row / sublanes) * tiles_per_row + (col / lanes)
offset     = tile_index * (sublanes * lanes) + (row % sublanes) * lanes + (col % lanes)
```

Tiles are visited row-major; within a tile, elements are row-major over
`(sublane, lane)`. This is precisely the order a TPU's vector memory uses, which is
what makes `as_storage_slice()` copy-ready. `Layout` precomputes the strides and
also provides the inverse map, which the sparsity and padding-fill paths rely on.

## The matmul simulator

`systolic` is a *reference model*, not a timing model. It reproduces the part of a
systolic array that affects results — the blocking into `mxu × mxu` weight loads and
the f32 accumulation across the contraction dimension — and nothing about cycles or
latency. Because the accumulation order matches the hardware, an `f32` result is
bit-identical to the device and a `bf16` result tracks it closely. The simulator
also reports padding MACs, so you can see the utilisation cost of an awkward shape
before you pay for it on real silicon.

## Sparsity at tile granularity

Element-level sparsity does not help a systolic array: it pays for a zero the same
as a one. The unit of savings is the *tile*. So `sparse` answers exactly one
question — which tiles are entirely zero — and lets a kernel skip them.

## Quantisation keeps the tiling

Int8 matmul is roughly four times the throughput of bf16, but only if you quantise
without destroying the layout. The `quantize` module maps an `f32` lattice to an
`i8` lattice element-by-element through the same `(row, col)` accessors, so the
hardware tiling survives the dtype change untouched.

## Non-goals

`systile` is deliberately *not* a general n-dimensional tensor library, an autograd
engine, or a device runtime. It is the host-side data structure and the reference
numerics. Keeping that scope small is what lets the layout stay honest to the
hardware.
