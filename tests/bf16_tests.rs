//! Tests for the `bf16` software float.

use systile::Bf16;

#[test]
fn zero_roundtrips() {
    assert_eq!(Bf16::from_f32(0.0).to_f32(), 0.0);
}

#[test]
fn one_roundtrips() {
    assert_eq!(Bf16::from_f32(1.0).to_f32(), 1.0);
}

#[test]
fn one_has_canonical_bits() {
    assert_eq!(Bf16::ONE.to_bits(), 0x3f80);
}

#[test]
fn negation_flips_sign_bit() {
    assert_eq!((-Bf16::ONE).to_bits(), 0xbf80);
}

#[test]
fn small_integers_are_exact() {
    for i in -64..=64 {
        let x = i as f32;
        assert_eq!(Bf16::from_f32(x).to_f32(), x, "i={i}");
    }
}

#[test]
fn powers_of_two_are_exact() {
    for e in -30..30 {
        let x = 2.0f32.powi(e);
        assert_eq!(Bf16::from_f32(x).to_f32(), x, "e={e}");
    }
}

#[test]
fn addition_matches_f32_for_exact_values() {
    let a = Bf16::from_f32(2.0);
    let b = Bf16::from_f32(3.0);
    assert_eq!((a + b).to_f32(), 5.0);
}

#[test]
fn subtraction_matches_f32_for_exact_values() {
    let a = Bf16::from_f32(5.0);
    let b = Bf16::from_f32(3.0);
    assert_eq!((a - b).to_f32(), 2.0);
}

#[test]
fn multiplication_matches_f32_for_exact_values() {
    let a = Bf16::from_f32(4.0);
    let b = Bf16::from_f32(0.5);
    assert_eq!((a * b).to_f32(), 2.0);
}

#[test]
fn division_matches_f32_for_exact_values() {
    let a = Bf16::from_f32(6.0);
    let b = Bf16::from_f32(2.0);
    assert_eq!((a / b).to_f32(), 3.0);
}

#[test]
fn add_assign_works() {
    let mut a = Bf16::from_f32(1.0);
    a += Bf16::from_f32(1.0);
    assert_eq!(a.to_f32(), 2.0);
}

#[test]
fn dynamic_range_survives_large() {
    assert!(Bf16::from_f32(1e30).is_finite());
}

#[test]
fn dynamic_range_survives_small() {
    assert!(Bf16::from_f32(1e-30).to_f32() > 0.0);
}

#[test]
fn nan_is_nan() {
    assert!(Bf16::from_f32(f32::NAN).is_nan());
}

#[test]
fn nan_never_equals_itself() {
    let n = Bf16::NAN;
    assert!(n != n);
}

#[test]
fn infinity_is_infinite() {
    assert!(Bf16::INFINITY.is_infinite());
    assert!(!Bf16::INFINITY.is_finite());
}

#[test]
fn positive_and_negative_zero_are_equal() {
    assert_eq!(Bf16::from_f32(0.0), Bf16::from_f32(-0.0));
}

#[test]
fn abs_clears_sign() {
    assert_eq!(Bf16::from_f32(-3.0).abs(), Bf16::from_f32(3.0));
}

#[test]
fn copysign_takes_sign_of_other() {
    let m = Bf16::from_f32(3.0).copysign(Bf16::from_f32(-1.0));
    assert!(m.is_sign_negative());
}

#[test]
fn ordering_is_sane() {
    assert!(Bf16::from_f32(1.0) < Bf16::from_f32(2.0));
    assert!(Bf16::from_f32(-1.0) < Bf16::from_f32(0.0));
}

#[test]
fn max_picks_larger() {
    assert_eq!(
        Bf16::from_f32(1.0).max(Bf16::from_f32(2.0)),
        Bf16::from_f32(2.0)
    );
}

#[test]
fn min_picks_smaller() {
    assert_eq!(
        Bf16::from_f32(1.0).min(Bf16::from_f32(2.0)),
        Bf16::from_f32(1.0)
    );
}

#[test]
fn sum_accumulates_in_f32() {
    let xs = [Bf16::from_f32(0.1); 10];
    let total: Bf16 = xs.into_iter().sum();
    assert!((total.to_f32() - 1.0).abs() < 0.05);
}

#[test]
fn default_is_zero() {
    assert_eq!(Bf16::default(), Bf16::ZERO);
}

#[test]
fn from_i8_works() {
    assert_eq!(Bf16::from(7i8).to_f32(), 7.0);
}

#[test]
fn clamp_bounds_value() {
    let lo = Bf16::from_f32(0.0);
    let hi = Bf16::from_f32(1.0);
    assert_eq!(Bf16::from_f32(2.0).clamp(lo, hi), hi);
    assert_eq!(Bf16::from_f32(-1.0).clamp(lo, hi), lo);
    assert_eq!(Bf16::from_f32(0.5).clamp(lo, hi).to_f32(), 0.5);
}

#[test]
fn recip_of_two_is_half() {
    assert_eq!(Bf16::from_f32(2.0).recip().to_f32(), 0.5);
}

#[test]
fn mul_add_matches_f32() {
    let r = Bf16::from_f32(2.0).mul_add(Bf16::from_f32(3.0), Bf16::from_f32(1.0));
    assert_eq!(r.to_f32(), 7.0);
}

#[test]
fn signum_reports_sign() {
    assert_eq!(Bf16::from_f32(5.0).signum(), Bf16::ONE);
    assert_eq!(Bf16::from_f32(-5.0).signum(), Bf16::NEG_ONE);
    assert!(Bf16::from_f32(0.0).signum().is_zero());
}

#[test]
fn one_is_normal() {
    assert!(Bf16::ONE.is_normal());
    assert!(!Bf16::ZERO.is_normal());
    assert!(!Bf16::INFINITY.is_normal());
}

#[test]
fn byte_roundtrip() {
    let b = Bf16::from_f32(3.5);
    assert_eq!(Bf16::from_le_bytes(b.to_le_bytes()), b);
}

