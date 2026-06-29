//! Tests for reductions over the logical region (padding must never leak in).

use systile::{Geometry, PaddedTileLattice};

fn sample() -> PaddedTileLattice<f32> {
    PaddedTileLattice::from_dense(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Geometry::TPU_V).unwrap()
}

#[test]
fn sum_ignores_padding() {
    assert_eq!(sample().sum(), 21.0);
}

#[test]
fn max_finds_largest() {
    assert_eq!(sample().max(), Some(6.0));
}

#[test]
fn min_finds_smallest() {
    assert_eq!(sample().min(), Some(1.0));
}

#[test]
fn mean_is_average() {
    assert_eq!(sample().mean(), Some(3.5));
}

#[test]
fn empty_max_is_none() {
    let l = PaddedTileLattice::<f32>::zeroed(0, 0, Geometry::TPU_V).unwrap();
    assert_eq!(l.max(), None);
}

#[test]
fn empty_mean_is_none() {
    let l = PaddedTileLattice::<f32>::zeroed(0, 0, Geometry::TPU_V).unwrap();
    assert_eq!(l.mean(), None);
}

