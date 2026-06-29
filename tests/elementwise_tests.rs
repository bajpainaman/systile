//! Tests for element-wise maps and binary combinators.

use systile::error::LatticeError;
use systile::{Geometry, PaddedTileLattice};

#[test]
fn map_applies_to_logical_only() {
    let l = PaddedTileLattice::from_dense(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Geometry::TPU_V)
        .unwrap();
    let doubled = l.map(|x| x * 2.0);
    assert_eq!(doubled.to_dense(), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn map_can_change_type() {
    let l = PaddedTileLattice::from_dense(1, 2, &[1.5f32, 2.5], Geometry::TPU_V).unwrap();
    let ints: PaddedTileLattice<i32> = l.map(|x| *x as i32);
    assert_eq!(ints.to_dense(), vec![1, 2]);
}

#[test]
fn map_in_place_mutates() {
    let mut l = PaddedTileLattice::from_dense(1, 3, &[1.0, 2.0, 3.0], Geometry::TPU_V).unwrap();
    l.map_in_place(|x| x + 1.0);
    assert_eq!(l.to_dense(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn zip_with_adds() {
    let a = PaddedTileLattice::from_dense(1, 3, &[1.0, 2.0, 3.0], Geometry::TPU_V).unwrap();
    let b = PaddedTileLattice::from_dense(1, 3, &[10.0, 20.0, 30.0], Geometry::TPU_V).unwrap();
    let c = a.zip_with(&b, |x, y| x + y).unwrap();
    assert_eq!(c.to_dense(), vec![11.0, 22.0, 33.0]);
}

#[test]
fn zip_with_shape_mismatch_errors() {
    let a = PaddedTileLattice::<f32>::zeroed(2, 3, Geometry::TPU_V).unwrap();
    let b = PaddedTileLattice::<f32>::zeroed(3, 2, Geometry::TPU_V).unwrap();
    assert!(matches!(
        a.zip_with(&b, |x, y| x + y),
        Err(LatticeError::ContractionMismatch { .. })
    ));
}

