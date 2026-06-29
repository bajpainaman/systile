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

#[test]
fn tile_len_is_product() {
    assert_eq!(Geometry::TPU_V.tile_len(), 8 * 128);
}

#[test]
fn zero_dimension_is_rejected() {
    assert_eq!(Geometry::new(0, 4, 4), Err(LatticeError::ZeroTileDimension));
}

