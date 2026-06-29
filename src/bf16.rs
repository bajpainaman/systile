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

    /// Negative one.
    pub const NEG_ONE: Bf16 = Bf16(0xbf80);

    /// Positive infinity.
    pub const INFINITY: Bf16 = Bf16(0x7f80);

    /// Negative infinity.
    pub const NEG_INFINITY: Bf16 = Bf16(0xff80);

    /// A quiet not-a-number.
    pub const NAN: Bf16 = Bf16(0x7fc0);

    /// The largest finite `bf16`.
    pub const MAX: Bf16 = Bf16(0x7f7f);

    /// The smallest (most negative) finite `bf16`.
    pub const MIN: Bf16 = Bf16(0xff7f);

    /// The smallest positive normal `bf16`.
    pub const MIN_POSITIVE: Bf16 = Bf16(0x0080);

    /// Construct a `bf16` directly from its raw 16-bit pattern.
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Bf16(bits)
    }

    /// Return the raw 16-bit pattern of this value.
    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert an `f32` to `bf16` using round-to-nearest-even.
    ///
    /// NaNs are preserved as quiet NaNs. The rounding bias is the standard
    /// "round half to even" used by hardware bf16 truncation units.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        if value.is_nan() {
            // Force a quiet NaN with a non-zero payload so it survives narrowing.
            return Bf16((bits >> 16) as u16 | 0x0040);
        }
        // Round to nearest even: add the rounding bias derived from the bit that
        // will be dropped plus the lsb of the kept mantissa.
        let rounding_bias = 0x7fff + ((bits >> 16) & 1);
        let rounded = bits.wrapping_add(rounding_bias);
        Bf16((rounded >> 16) as u16)
    }

    /// Convert this `bf16` back to an `f32` exactly (the low 16 bits are zero).
    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// True if this value is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7f80) == 0x7f80 && (self.0 & 0x007f) != 0
    }

    /// True if this value is positive or negative infinity.
    #[inline]
    pub fn is_infinite(self) -> bool {
        (self.0 & 0x7fff) == 0x7f80
    }

