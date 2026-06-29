//! Tests for lattice construction, access, and dense round-trips.

use systile::error::LatticeError;
use systile::{Geometry, PaddedTileLattice};

fn sample() -> PaddedTileLattice<f32> {
    PaddedTileLattice::from_dense(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Geometry::TPU_V).unwrap()
}

#[test]
fn from_dense_preserves_dims() {
    let l = sample();
    assert_eq!(l.rows(), 2);
    assert_eq!(l.cols(), 3);
}

#[test]
fn dense_roundtrip_is_identity() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let l = PaddedTileLattice::from_dense(2, 3, &data, Geometry::TPU_V).unwrap();
    assert_eq!(l.to_dense(), data.to_vec());
}

#[test]
fn wrong_buffer_length_is_rejected() {
    let err = PaddedTileLattice::from_dense(2, 3, &[1.0, 2.0], Geometry::TPU_V).unwrap_err();
    assert_eq!(
        err,
        LatticeError::BufferLengthMismatch {
            expected: 6,
            actual: 2
        }
    );
}

#[test]
fn get_returns_logical_values() {
    let l = sample();
    assert_eq!(l.get(0, 0), Some(&1.0));
    assert_eq!(l.get(1, 2), Some(&6.0));
}

#[test]
fn get_out_of_bounds_is_none() {
    let l = sample();
    assert_eq!(l.get(2, 0), None);
    assert_eq!(l.get(0, 3), None);
}

#[test]
fn set_updates_value() {
    let mut l = sample();
    l.set(0, 0, 99.0).unwrap();
    assert_eq!(l.get(0, 0), Some(&99.0));
}

#[test]
fn set_out_of_bounds_errors() {
    let mut l = sample();
    assert_eq!(
        l.set(5, 5, 0.0),
        Err(LatticeError::IndexOutOfBounds { row: 5, col: 5 })
    );
}

#[test]
fn get_mut_allows_mutation() {
    let mut l = sample();
    *l.get_mut(1, 1).unwrap() = 42.0;
    assert_eq!(l.get(1, 1), Some(&42.0));
}

#[test]
fn zeroed_starts_empty() {
    let l = PaddedTileLattice::<f32>::zeroed(4, 4, Geometry::TPU_V).unwrap();
    assert_eq!(l.to_dense(), vec![0.0; 16]);
}

#[test]
fn padded_len_includes_padding() {
    let l = sample();
    assert_eq!(l.padded_len(), 8 * 128);
}

#[test]
fn one_small_matrix_is_a_single_tile() {
    let l = sample();
    assert_eq!(l.num_tiles(), 1);
}

#[test]
fn storage_slice_is_padded_length() {
    let l = sample();
    assert_eq!(l.as_storage_slice().len(), l.padded_len());
}

#[test]
fn len_is_logical_product() {
    assert_eq!(sample().len(), 6);
}

#[test]
fn is_empty_only_when_no_elements() {
    assert!(!sample().is_empty());
    assert!(PaddedTileLattice::<f32>::zeroed(0, 0, Geometry::TPU_V)
        .unwrap()
        .is_empty());
}

#[test]
fn fill_padding_does_not_touch_logical() {
    let mut l = sample();
    l.fill_padding(-1.0);
    assert_eq!(l.to_dense(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn fill_padding_changes_padding_slots() {
    let mut l = sample();
    l.fill_padding(-1.0);
    // The slot just past the last logical column is padding.
    let off = l.layout().offset(0, 3);
    assert_eq!(l.as_storage_slice()[off], -1.0);
}

#[test]
fn multi_tile_shape_has_many_tiles() {
    let l = PaddedTileLattice::<f32>::zeroed(130, 257, Geometry::TPU_V).unwrap();
    // 130 -> 17 row tiles (ceil 130/8 = 17), 257 -> 3 col tiles.
    assert_eq!(l.num_tiles(), 17 * 3);
}

