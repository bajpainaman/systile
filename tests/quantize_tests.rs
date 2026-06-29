//! Tests for int8 affine quantisation.

use systile::{Geometry, PaddedTileLattice, QuantParams};

#[test]
fn symmetric_zero_point_is_zero() {
    assert_eq!(QuantParams::symmetric(1.0).zero_point, 0);
}

