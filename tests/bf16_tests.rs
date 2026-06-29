//! Tests for the `bf16` software float.

use systile::Bf16;

#[test]
fn zero_roundtrips() {
    assert_eq!(Bf16::from_f32(0.0).to_f32(), 0.0);
}

