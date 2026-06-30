//! Tests for the matmul nearest-neighbour index.

use systile::TensorIndex;

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rvec(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| (splitmix64(seed.wrapping_add(i as u64)) as f32 / u64::MAX as f32) * 2.0 - 1.0)
        .collect()
}

#[test]
fn empty_index() {
    let idx = TensorIndex::new(16);
    assert!(idx.is_empty());
    assert_eq!(idx.nearest(&[0.0; 16]), None);
}

#[test]
fn add_assigns_sequential_ids() {
    let mut idx = TensorIndex::new(4);
    assert_eq!(idx.add(vec![1.0, 0.0, 0.0, 0.0]), 0);
    assert_eq!(idx.add(vec![0.0, 1.0, 0.0, 0.0]), 1);
    assert_eq!(idx.len(), 2);
}

#[test]
fn nearest_is_self_for_stored_vector() {
    let mut idx = TensorIndex::new(64);
    for j in 0..50 {
        idx.add(rvec(64, j));
    }
    for j in 0..50usize {
        let q = idx.vector(j).to_vec();
        assert_eq!(idx.nearest(&q).unwrap().id, j, "item {j}");
    }
}

#[test]
fn noisy_query_recovers_target() {
    let dim = 128;
    let mut idx = TensorIndex::new(dim);
    for j in 0..500 {
        idx.add(rvec(dim, 0x10 + j as u64));
    }
    let target = 321;
    let mut q = idx.vector(target).to_vec();
    for (i, x) in q.iter_mut().enumerate() {
        if i % 5 == 0 {
            *x += 0.03;
        }
    }
    assert_eq!(idx.nearest(&q).unwrap().id, target);
}

#[test]
fn search_returns_k_sorted_descending() {
    let dim = 32;
    let mut idx = TensorIndex::new(dim);
    for j in 0..20 {
        idx.add(rvec(dim, j));
    }
    let hits = idx.search(idx.vector(3), 5);
    assert_eq!(hits.len(), 5);
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score);
    }
    assert_eq!(hits[0].id, 3);
}

#[test]
fn batch_search_matches_single() {
    let dim = 48;
    let mut idx = TensorIndex::new(dim);
    for j in 0..40 {
        idx.add(rvec(dim, 0x55 + j as u64));
    }
    let queries: Vec<Vec<f32>> = (0..5).map(|j| idx.vector(j).to_vec()).collect();
    let refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();
    let batch = idx.search_batch(&refs, 3);
    for (j, hits) in batch.iter().enumerate() {
        assert_eq!(hits[0].id, idx.search(&queries[j], 3)[0].id);
    }
}

#[test]
fn score_batch_has_n_columns() {
    let dim = 16;
    let mut idx = TensorIndex::new(dim);
    for j in 0..7 {
        idx.add(rvec(dim, j));
    }
    let scores = idx.score_batch(&[idx.vector(0)]);
    assert_eq!(scores[0].len(), 7);
}
