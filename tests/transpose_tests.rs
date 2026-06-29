//! Tests for transpose and relayout.

use systile::{Geometry, PaddedTileLattice};

#[test]
fn transpose_swaps_dims() {
    let l = PaddedTileLattice::from_dense(2, 3, &[1.0; 6], Geometry::TPU_V).unwrap();
    let t = l.transpose();
    assert_eq!(t.rows(), 3);
    assert_eq!(t.cols(), 2);
}

#[test]
fn transpose_swaps_elements() {
    let l = PaddedTileLattice::from_dense(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Geometry::TPU_V)
        .unwrap();
    let t = l.transpose();
    assert_eq!(t.get(0, 0), Some(&1.0));
    assert_eq!(t.get(2, 1), Some(&6.0));
    assert_eq!(t.get(1, 0), Some(&2.0));
}

#[test]
fn double_transpose_is_identity() {
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let l = PaddedTileLattice::from_dense(3, 4, &data, Geometry::TPU_V).unwrap();
    assert_eq!(l.transpose().transpose().to_dense(), data);
}

#[test]
fn transpose_is_involutive_helper() {
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let l = PaddedTileLattice::from_dense(3, 4, &data, Geometry::TPU_V).unwrap();
    assert!(l.is_transpose_involutive());
}

#[test]
fn relayout_preserves_logical_data() {
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let l = PaddedTileLattice::from_dense(3, 4, &data, Geometry::TPU_V).unwrap();
    let r = l.relayout(Geometry::TINY).unwrap();
    assert_eq!(r.to_dense(), data);
}

