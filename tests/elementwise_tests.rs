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

