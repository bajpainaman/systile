//! `bf16` — the brain-floating-point format that is the native compute dtype of a TPU.
//!
//! A `bf16` is the top 16 bits of an IEEE-754 `f32`: one sign bit, eight exponent bits,
//! and seven mantissa bits. It keeps the full `f32` dynamic range while throwing away
//! mantissa precision, which is exactly the trade a systolic matrix unit wants.
//!
//! This is a from-scratch software implementation. All arithmetic is performed by
//! widening to `f32`, computing in `f32`, and narrowing back with round-to-nearest-even.
//! That mirrors how a TPU accumulates bf16 products into an f32 accumulator.

use core::cmp::Ordering;
use core::fmt;
use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A brain floating point number (bfloat16).
///
/// The value is stored as the raw 16 bits that would occupy the high half of the
/// corresponding `f32`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Bf16(u16);

impl Bf16 {
    /// Positive zero.
    pub const ZERO: Bf16 = Bf16(0x0000);

    /// One.
    pub const ONE: Bf16 = Bf16(0x3f80);

