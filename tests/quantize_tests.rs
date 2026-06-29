//! Tests for int8 affine quantisation.

use systile::{Geometry, PaddedTileLattice, QuantParams};

#[test]
fn symmetric_zero_point_is_zero() {
    assert_eq!(QuantParams::symmetric(1.0).zero_point, 0);
}

#[test]
fn symmetric_max_maps_to_127() {
    let q = QuantParams::symmetric(1.0);
    assert_eq!(q.quantize(1.0), 127);
}

#[test]
fn symmetric_negation_is_representable() {
    let q = QuantParams::symmetric(1.0);
    assert_eq!(q.quantize(-1.0), -127);
}

#[test]
fn symmetric_zero_input_handled() {
    let q = QuantParams::symmetric(0.0);
    assert_eq!(q.quantize(0.0), 0);
}

