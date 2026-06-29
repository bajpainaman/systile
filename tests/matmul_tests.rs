//! Tests for the systolic matmul simulator. Correctness is checked against a
//! naive triple loop so the tiled dataflow can never silently diverge.

use systile::error::LatticeError;
use systile::{Bf16, Geometry, PaddedTileLattice};

fn naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

#[test]
fn small_square_matches_naive() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let la = PaddedTileLattice::from_dense(2, 2, &a, Geometry::TPU_V).unwrap();
    let lb = PaddedTileLattice::from_dense(2, 2, &b, Geometry::TPU_V).unwrap();
    let c = la.matmul(&lb).unwrap();
    assert_eq!(c.to_dense(), naive(&a, &b, 2, 2, 2));
}

#[test]
fn identity_is_neutral() {
    let mut id = PaddedTileLattice::<f32>::zeroed(4, 4, Geometry::TPU_V).unwrap();
    for i in 0..4 {
        id.set(i, i, 1.0).unwrap();
    }
    let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let m = PaddedTileLattice::from_dense(4, 4, &data, Geometry::TPU_V).unwrap();
    let c = m.matmul(&id).unwrap();
    assert_eq!(c.to_dense(), data);
}

#[test]
fn rectangular_matches_naive() {
    let a: Vec<f32> = (0..6).map(|x| x as f32).collect();
    let b: Vec<f32> = (0..6).map(|x| (x + 1) as f32).collect();
    let la = PaddedTileLattice::from_dense(2, 3, &a, Geometry::TPU_V).unwrap();
    let lb = PaddedTileLattice::from_dense(3, 2, &b, Geometry::TPU_V).unwrap();
    let c = la.matmul(&lb).unwrap();
    assert_eq!(c.to_dense(), naive(&a, &b, 2, 3, 2));
}

#[test]
fn multi_mxu_block_matches_naive() {
    // Use the tiny geometry so a 6x6 spans several mxu blocks.
    let a: Vec<f32> = (0..36).map(|x| (x % 5) as f32).collect();
    let b: Vec<f32> = (0..36).map(|x| (x % 3) as f32).collect();
    let la = PaddedTileLattice::from_dense(6, 6, &a, Geometry::TINY).unwrap();
    let lb = PaddedTileLattice::from_dense(6, 6, &b, Geometry::TINY).unwrap();
    let c = la.matmul(&lb).unwrap();
    assert_eq!(c.to_dense(), naive(&a, &b, 6, 6, 6));
}

#[test]
fn contraction_mismatch_is_rejected() {
    let a = PaddedTileLattice::<f32>::zeroed(2, 3, Geometry::TPU_V).unwrap();
    let b = PaddedTileLattice::<f32>::zeroed(4, 2, Geometry::TPU_V).unwrap();
    assert_eq!(
        a.matmul(&b).unwrap_err(),
        LatticeError::ContractionMismatch {
            lhs_cols: 3,
            rhs_rows: 4
        }
    );
}

#[test]
fn geometry_mismatch_is_rejected() {
    let a = PaddedTileLattice::<f32>::zeroed(2, 2, Geometry::TPU_V).unwrap();
    let b = PaddedTileLattice::<f32>::zeroed(2, 2, Geometry::TINY).unwrap();
    assert_eq!(a.matmul(&b).unwrap_err(), LatticeError::GeometryMismatch);
}

