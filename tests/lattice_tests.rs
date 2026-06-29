//! Tests for lattice construction, access, and dense round-trips.

use systile::error::LatticeError;
use systile::{Geometry, PaddedTileLattice};

fn sample() -> PaddedTileLattice<f32> {
    PaddedTileLattice::from_dense(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Geometry::TPU_V).unwrap()
}

