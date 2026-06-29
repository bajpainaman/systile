//! Tests for transpose and relayout.

use systile::{Geometry, PaddedTileLattice};

#[test]
fn transpose_swaps_dims() {
    let l = PaddedTileLattice::from_dense(2, 3, &[1.0; 6], Geometry::TPU_V).unwrap();
    let t = l.transpose();
    assert_eq!(t.rows(), 3);
    assert_eq!(t.cols(), 2);
}

