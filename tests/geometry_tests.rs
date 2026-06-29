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

#[test]
fn pad_cols_uses_lanes() {
    assert_eq!(Geometry::TPU_V.pad_cols(1), 128);
    assert_eq!(Geometry::TPU_V.pad_cols(129), 256);
}

#[test]
fn tile_counts_are_correct() {
    assert_eq!(Geometry::TPU_V.tile_rows(9), 2);
    assert_eq!(Geometry::TPU_V.tile_cols(129), 2);
}

#[test]
fn alignment_detection() {
    assert!(Geometry::TPU_V.is_aligned(8, 128));
    assert!(!Geometry::TPU_V.is_aligned(3, 5));
}

#[test]
fn default_geometry_is_tpu_v() {
    assert_eq!(Geometry::default(), Geometry::TPU_V);
}

#[test]
fn shape_tracks_padding() {
    let s = Shape::new(3, 5, &Geometry::TPU_V);
    assert_eq!(s.rows, 3);
    assert_eq!(s.cols, 5);
    assert_eq!(s.padded_rows, 8);
    assert_eq!(s.padded_cols, 128);
}

#[test]
fn shape_logical_len() {
    assert_eq!(Shape::new(3, 5, &Geometry::TPU_V).logical_len(), 15);
}

#[test]
fn shape_padded_len() {
    assert_eq!(Shape::new(3, 5, &Geometry::TPU_V).padded_len(), 8 * 128);
}

#[test]
fn shape_padding_len() {
    let s = Shape::new(3, 5, &Geometry::TPU_V);
    assert_eq!(s.padding_len(), 8 * 128 - 15);
}

#[test]
fn exact_shape_has_no_padding() {
    let s = Shape::new(8, 128, &Geometry::TPU_V);
    assert!(s.is_exact());
    assert_eq!(s.padding_len(), 0);
}

