//! Tests for the triangular-matmul prefix scan.

use systile::TensorScan;

fn cumsum(x: &[f32]) -> Vec<f32> {
    let mut acc = 0.0;
    x.iter()
        .map(|&v| {
            acc += v;
            acc
        })
        .collect()
}

#[test]
fn inclusive_matches_cumsum() {
    let s = TensorScan::new();
    let x = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0];
    assert_eq!(s.inclusive(&x), cumsum(&x));
}

#[test]
fn exclusive_starts_at_zero() {
    let s = TensorScan::new();
    let x = [2.0f32, 4.0, 6.0];
    assert_eq!(s.exclusive(&x), vec![0.0, 2.0, 6.0]);
}

#[test]
fn inclusive_minus_exclusive_is_input() {
    let s = TensorScan::new();
    let x = [5.0f32, 2.0, 7.0, 1.0];
    let inc = s.inclusive(&x);
    let exc = s.exclusive(&x);
    for i in 0..x.len() {
        assert!((inc[i] - exc[i] - x[i]).abs() < 1e-4);
    }
}

#[test]
fn suffix_is_reverse_prefix() {
    let s = TensorScan::new();
    let x = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(s.suffix(&x), vec![10.0, 9.0, 7.0, 4.0]);
}

#[test]
fn total_is_sum() {
    let s = TensorScan::new();
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(s.total(&x), 15.0);
}

#[test]
fn empty_input() {
    let s = TensorScan::new();
    assert!(s.inclusive(&[]).is_empty());
    assert_eq!(s.total(&[]), 0.0);
}

#[test]
fn single_element() {
    let s = TensorScan::new();
    assert_eq!(s.inclusive(&[7.0]), vec![7.0]);
    assert_eq!(s.exclusive(&[7.0]), vec![0.0]);
}
