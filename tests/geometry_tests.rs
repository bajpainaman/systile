//! Tests for tile geometry and padding arithmetic.

use systile::error::LatticeError;
use systile::{Geometry, Shape};

#[test]
fn tpu_v_geometry_dims() {
    let g = Geometry::TPU_V;
    assert_eq!(g.sublanes, 8);
    assert_eq!(g.lanes, 128);
    assert_eq!(g.mxu, 128);
}

