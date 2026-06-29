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

#[test]
fn quantize_dequantize_is_close() {
    let q = QuantParams::symmetric(4.0);
    for i in -16..=16 {
        let x = i as f32 * 0.25;
        let back = q.dequantize(q.quantize(x));
        assert!((x - back).abs() <= q.scale, "x={x} back={back}");
    }
}

#[test]
fn quantize_saturates_high() {
    let q = QuantParams::symmetric(1.0);
    assert_eq!(q.quantize(1000.0), 127);
}

#[test]
fn quantize_saturates_low() {
    let q = QuantParams::symmetric(1.0);
    assert_eq!(q.quantize(-1000.0), -128);
}

#[test]
fn asymmetric_maps_min_near_floor() {
    let q = QuantParams::asymmetric(-3.0, 5.0);
    let back = q.dequantize(q.quantize(-3.0));
    assert!((back - (-3.0)).abs() < q.scale * 2.0);
}

#[test]
fn lattice_abs_max_finds_extreme() {
    let l = PaddedTileLattice::from_dense(2, 2, &[-7.0, 1.0, 2.0, 3.0], Geometry::TPU_V).unwrap();
    assert_eq!(l.abs_max(), 7.0);
}

