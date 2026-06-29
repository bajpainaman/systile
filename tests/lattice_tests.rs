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

