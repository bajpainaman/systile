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

