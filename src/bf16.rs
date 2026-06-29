//! `bf16` — the brain-floating-point format that is the native compute dtype of a TPU.
//!
//! A `bf16` is the top 16 bits of an IEEE-754 `f32`: one sign bit, eight exponent bits,
//! and seven mantissa bits. It keeps the full `f32` dynamic range while throwing away
//! mantissa precision, which is exactly the trade a systolic matrix unit wants.
//!
//! This is a from-scratch software implementation. All arithmetic is performed by
//! widening to `f32`, computing in `f32`, and narrowing back with round-to-nearest-even.
//! That mirrors how a TPU accumulates bf16 products into an f32 accumulator.

