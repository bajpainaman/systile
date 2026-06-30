//! Tests for the comparison-matmul sorter.

use systile::TensorSort;

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn sorts_a_small_vector() {
    let s = TensorSort::new();
    assert_eq!(s.sort(&[3.0, 1.0, 2.0]), vec![1.0, 2.0, 3.0]);
}

#[test]
fn ranks_are_a_permutation() {
    let s = TensorSort::new();
    let x = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let mut ranks = s.ranks(&x);
    ranks.sort_unstable();
    assert_eq!(ranks, (0..x.len()).collect::<Vec<_>>());
}

#[test]
fn handles_ties_stably() {
    let s = TensorSort::new();
    // Two equal values must occupy adjacent, distinct ranks.
    let x = [2.0f32, 1.0, 2.0, 1.0];
    let ranks = s.ranks(&x);
    let mut seen = ranks.clone();
    seen.sort_unstable();
    assert_eq!(seen, vec![0, 1, 2, 3]);
    assert_eq!(s.sort(&x), vec![1.0, 1.0, 2.0, 2.0]);
}

#[test]
fn argsort_matches_std() {
    let s = TensorSort::new();
    let x = [5.0f32, 3.0, 8.0, 1.0, 9.0, 2.0];
    let got = s.argsort(&x);
    let mut expected: Vec<usize> = (0..x.len()).collect();
    expected.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap().then(a.cmp(&b)));
    assert_eq!(got, expected);
}

#[test]
fn permutation_matmul_sorts() {
    let s = TensorSort::new();
    let x = [4.0f32, 2.0, 7.0, 1.0, 3.0];
    assert_eq!(s.sort_via_matmul(&x), s.sort(&x));
}

#[test]
fn matches_std_sort_on_random_inputs() {
    let s = TensorSort::new();
    for trial in 0..20u64 {
        let n = 5 + (trial as usize % 20);
        let x: Vec<f32> = (0..n)
            .map(|i| (splitmix64(trial * 100 + i as u64) % 1000) as f32)
            .collect();
        let mut expected = x.clone();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(s.sort(&x), expected, "trial {trial}");
    }
}

#[test]
fn already_sorted_is_identity() {
    let s = TensorSort::new();
    let x = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(s.argsort(&x), vec![0, 1, 2, 3]);
}

#[test]
fn empty_input() {
    let s = TensorSort::new();
    assert!(s.sort(&[]).is_empty());
    assert!(s.argsort(&[]).is_empty());
}
