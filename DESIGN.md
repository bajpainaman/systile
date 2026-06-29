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

