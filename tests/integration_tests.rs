//! End-to-end tests that exercise several features together, the way a real
//! caller would chain them.

use systile::prelude::*;

fn ramp(rows: usize, cols: usize) -> PaddedTileLattice<f32> {
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    PaddedTileLattice::from_dense(rows, cols, &data, Geometry::TPU_V).unwrap()
}

#[test]
fn dense_roundtrip_through_geometry_relayout() {
    let l = ramp(5, 7);
    let original = l.to_dense();
    let back = l
        .relayout(Geometry::TINY)
        .unwrap()
        .relayout(Geometry::TPU_V)
        .unwrap();
    assert_eq!(back.to_dense(), original);
}

#[test]
fn transpose_then_matmul_is_gram_matrix() {
    let a = ramp(2, 3);
    let g = a.transpose().matmul(&a).unwrap();
    assert_eq!(g.rows(), 3);
    assert_eq!(g.cols(), 3);
}

