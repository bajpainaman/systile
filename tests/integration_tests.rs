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

#[test]
fn gram_matrix_is_symmetric() {
    let a = ramp(2, 3);
    let g = a.transpose().matmul(&a).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(g.get(i, j), g.get(j, i));
        }
    }
}

#[test]
fn quantize_roundtrip_preserves_sign_pattern() {
    let data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0)).collect();
    let l = PaddedTileLattice::from_dense(4, 4, &data, Geometry::TPU_V).unwrap();
    let params = QuantParams::symmetric(l.abs_max());
    let back = l.quantize(params).unwrap().dequantize(params).unwrap();
    for (orig, got) in l.to_dense().iter().zip(back.to_dense().iter()) {
        assert_eq!(
            orig.signum() as i32,
            got.signum() as i32,
            "orig={orig} got={got}"
        );
    }
}

#[test]
fn sparsity_survives_relayout() {
    let mut l = PaddedTileLattice::<f32>::zeroed(16, 16, Geometry::TPU_V).unwrap();
    l.set(0, 0, 1.0).unwrap();
    let dense_nonzero = l.to_dense().iter().filter(|x| **x != 0.0).count();
    let r = l.relayout(Geometry::TINY).unwrap();
    let dense_nonzero_after = r.to_dense().iter().filter(|x| **x != 0.0).count();
    assert_eq!(dense_nonzero, dense_nonzero_after);
}

#[test]
fn map_then_reduce_composes() {
    let l = ramp(2, 3);
    let squared = l.map(|x| x * x);
    assert_eq!(squared.sum(), (0..6).map(|i| (i * i) as f32).sum());
}

