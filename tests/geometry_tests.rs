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

#[test]
fn valid_geometry_is_accepted() {
    assert!(Geometry::new(8, 128, 128).is_ok());
}

#[test]
fn round_up_rounds_to_multiple() {
    assert_eq!(Geometry::round_up(1, 8), 8);
    assert_eq!(Geometry::round_up(8, 8), 8);
    assert_eq!(Geometry::round_up(9, 8), 16);
}

#[test]
fn pad_rows_uses_sublanes() {
    assert_eq!(Geometry::TPU_V.pad_rows(3), 8);
    assert_eq!(Geometry::TPU_V.pad_rows(8), 8);
    assert_eq!(Geometry::TPU_V.pad_rows(9), 16);
}

