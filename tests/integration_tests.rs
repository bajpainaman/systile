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

#[test]
fn zip_add_matches_elementwise_sum() {
    let a = ramp(3, 3);
    let b = ramp(3, 3);
    let c = a.zip_with(&b, |x, y| x + y).unwrap();
    for (got, base) in c.to_dense().iter().zip(a.to_dense().iter()) {
        assert_eq!(*got, base * 2.0);
    }
}

#[test]
fn bf16_lattice_roundtrips_through_dense() {
    let data: Vec<Bf16> = (0..6).map(|i| Bf16::from_f32(i as f32)).collect();
    let l = PaddedTileLattice::from_dense(2, 3, &data, Geometry::TPU_V).unwrap();
    assert_eq!(l.to_dense(), data);
}

#[test]
fn f32_to_bf16_lattice_via_map() {
    let l = ramp(2, 3);
    let b: PaddedTileLattice<Bf16> = l.map(|x| Bf16::from_f32(*x));
    assert_eq!(b.get(1, 2).unwrap().to_f32(), 5.0);
}

#[test]
fn matmul_associativity_holds_on_small_f32() {
    let a = ramp(2, 2);
    let b = ramp(2, 2);
    let c = ramp(2, 2);
    let left = a.matmul(&b).unwrap().matmul(&c).unwrap();
    let right = a.matmul(&b.matmul(&c).unwrap()).unwrap();
    assert_eq!(left.to_dense(), right.to_dense());
}

#[test]
fn padding_fill_does_not_affect_matmul() {
    let mut a = ramp(2, 3);
    let b = ramp(3, 2);
    let clean = a.matmul(&b).unwrap().to_dense();
    a.fill_padding(999.0);
    let filled = a.matmul(&b).unwrap().to_dense();
    assert_eq!(clean, filled);
}

#[test]
fn storage_slice_length_is_stable_across_set() {
    let mut l = ramp(3, 5);
    let before = l.as_storage_slice().len();
    l.set(0, 0, 42.0).unwrap();
    assert_eq!(l.as_storage_slice().len(), before);
}

#[test]
fn num_tiles_matches_iter_count_for_many_shapes() {
    for &(r, c) in &[(1, 1), (8, 128), (9, 129), (130, 257)] {
        let l = PaddedTileLattice::<f32>::zeroed(r, c, Geometry::TPU_V).unwrap();
        assert_eq!(l.num_tiles(), l.iter_tiles().count());
    }
}

#[test]
fn identity_matmul_via_quantized_path_is_close() {
    let mut id = PaddedTileLattice::<f32>::zeroed(4, 4, Geometry::TPU_V).unwrap();
    for i in 0..4 {
        id.set(i, i, 1.0).unwrap();
    }
    let m = ramp(4, 4);
    let params = QuantParams::symmetric(m.abs_max());
    let mq = m.quantize(params).unwrap().dequantize(params).unwrap();
    let out = mq.matmul(&id).unwrap();
    for (got, want) in out.to_dense().iter().zip(m.to_dense().iter()) {
        assert!(
            (got - want).abs() <= params.scale + 1e-3,
            "got={got} want={want}"
        );
    }
}

#[test]
fn row_sums_equal_matmul_with_ones() {
    let a = ramp(3, 4);
    let ones = PaddedTileLattice::from_dense(4, 1, &[1.0; 4], Geometry::TPU_V).unwrap();
    let prod = a.matmul(&ones).unwrap();
    for (i, sum) in a.row_sums().iter().enumerate() {
        assert_eq!(prod.get(i, 0).unwrap(), sum);
    }
}

#[test]
fn tile_density_plus_sparsity_is_one_after_quantize() {
    let mut l = PaddedTileLattice::<f32>::zeroed(16, 16, Geometry::TPU_V).unwrap();
    l.set(3, 3, 5.0).unwrap();
    let q = l.quantize(QuantParams::symmetric(5.0)).unwrap();
    assert!((q.tile_density() + q.tile_sparsity() - 1.0).abs() < 1e-9);
}

