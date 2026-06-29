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

