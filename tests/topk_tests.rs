//! Tests for the comparison-matmul top-k selector.

use systile::TensorTopK;

#[test]
fn ranks_are_descending_positions() {
    let t = TensorTopK::new();
    // largest gets rank 0.
    assert_eq!(t.ranks(&[1.0, 3.0, 2.0]), vec![2, 0, 1]);
}

#[test]
fn select_returns_k_best() {
    let t = TensorTopK::new();
    let scores = [0.2f32, 0.9, 0.5, 0.95, 0.1, 0.7];
    assert_eq!(t.select_indices(&scores, 2), vec![3, 1]);
}

#[test]
fn select_is_ordered_best_first() {
    let t = TensorTopK::new();
    let scores = [5.0f32, 1.0, 9.0, 3.0, 7.0];
    let sel = t.select(&scores, 3);
    assert_eq!(sel[0].1, 9.0);
    assert_eq!(sel[1].1, 7.0);
    assert_eq!(sel[2].1, 5.0);
}

#[test]
fn kth_largest_is_correct() {
    let t = TensorTopK::new();
    let scores = [5.0f32, 1.0, 9.0, 3.0, 7.0];
    assert_eq!(t.kth_largest(&scores, 1), Some(9.0));
    assert_eq!(t.kth_largest(&scores, 3), Some(5.0));
    assert_eq!(t.kth_largest(&scores, 6), None);
}

#[test]
fn matches_sorted_topk() {
    let t = TensorTopK::new();
    let scores = [3.0f32, 8.0, 1.0, 9.0, 4.0, 7.0, 2.0, 6.0];
    let got: Vec<f32> = t.select(&scores, 4).into_iter().map(|(_, s)| s).collect();
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(got, sorted[..4].to_vec());
}

#[test]
fn k_larger_than_len_returns_all() {
    let t = TensorTopK::new();
    let scores = [2.0f32, 1.0, 3.0];
    assert_eq!(t.select(&scores, 10).len(), 3);
}

#[test]
fn batch_selects_per_row() {
    let t = TensorTopK::new();
    let a = [1.0f32, 5.0, 2.0];
    let b = [9.0f32, 3.0, 4.0];
    let rows: Vec<&[f32]> = vec![&a, &b];
    let out = t.select_batch(&rows, 1);
    assert_eq!(out[0][0].0, 1);
    assert_eq!(out[1][0].0, 0);
}

#[test]
fn empty_input() {
    let t = TensorTopK::new();
    assert!(t.select(&[], 3).is_empty());
}
