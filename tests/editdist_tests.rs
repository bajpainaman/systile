//! Tests for tropical-matmul edit distance, checked against a reference DP.

use systile::TensorEditDistance;

/// Standard Levenshtein DP, used as ground truth.
fn reference(a: &[u8], b: &[u8]) -> usize {
    let (m, n) = (a.len(), b.len());
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut cur = vec![0usize; n + 1];
    for i in 1..=m {
        cur[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[n]
}

#[test]
fn classic_kitten_sitting() {
    let ed = TensorEditDistance::new();
    assert_eq!(ed.distance_str("kitten", "sitting"), 3);
}

#[test]
fn identical_strings_are_zero() {
    let ed = TensorEditDistance::new();
    assert_eq!(ed.distance_str("systile", "systile"), 0);
}

#[test]
fn empty_to_string_is_length() {
    let ed = TensorEditDistance::new();
    assert_eq!(ed.distance_str("", "abcd"), 4);
    assert_eq!(ed.distance_str("xyz", ""), 3);
}

#[test]
fn single_edits() {
    let ed = TensorEditDistance::new();
    assert_eq!(ed.distance_str("cat", "cap"), 1); // substitute
    assert_eq!(ed.distance_str("cat", "cats"), 1); // insert
    assert_eq!(ed.distance_str("cats", "cat"), 1); // delete
}

#[test]
fn matches_reference_on_many_pairs() {
    let ed = TensorEditDistance::new();
    let words = [
        "flaw", "lawn", "gumbo", "gambol", "book", "back", "abcd", "dcba",
    ];
    for &a in &words {
        for &b in &words {
            assert_eq!(
                ed.distance_str(a, b),
                reference(a.as_bytes(), b.as_bytes()),
                "{a} -> {b}"
            );
        }
    }
}

#[test]
fn distance_is_symmetric() {
    let ed = TensorEditDistance::new();
    assert_eq!(
        ed.distance_str("sunday", "saturday"),
        ed.distance_str("saturday", "sunday")
    );
}
