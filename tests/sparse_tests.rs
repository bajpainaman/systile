//! Tests for tile-level sparsity detection.

use systile::{Geometry, PaddedTileLattice};

#[test]
fn all_zero_tile_is_detected() {
    let l = PaddedTileLattice::<f32>::zeroed(2, 2, Geometry::TINY).unwrap();
    assert!(l.is_tile_zero(0, 0));
}

#[test]
fn nonzero_tile_is_not_zero() {
    let mut l = PaddedTileLattice::<f32>::zeroed(2, 2, Geometry::TINY).unwrap();
    l.set(0, 0, 1.0).unwrap();
    assert!(!l.is_tile_zero(0, 0));
}

