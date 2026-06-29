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

