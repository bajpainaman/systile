//! Tests for tile-level sparsity detection.

use systile::{Geometry, PaddedTileLattice};

#[test]
fn all_zero_tile_is_detected() {
    let l = PaddedTileLattice::<f32>::zeroed(2, 2, Geometry::TINY).unwrap();
    assert!(l.is_tile_zero(0, 0));
}

