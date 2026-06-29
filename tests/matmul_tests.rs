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

