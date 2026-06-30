//! Tests for the matmul Bloom filter.

use systile::TensorBloom;

#[test]
fn empty_filter_contains_nothing() {
    let b = TensorBloom::new(1024, 4, 1);
    assert!(!b.contains(42));
    assert_eq!(b.inserted(), 0);
}

#[test]
fn inserted_item_is_present() {
    let mut b = TensorBloom::new(1024, 4, 1);
    b.insert(42);
    assert!(b.contains(42));
}

#[test]
fn no_false_negatives() {
    let mut b = TensorBloom::new(4096, 5, 7);
    for i in 0..500u64 {
        b.insert(i * 3);
    }
    for i in 0..500u64 {
        assert!(b.contains(i * 3), "missed {}", i * 3);
    }
}

#[test]
fn batch_matches_scalar() {
    let mut b = TensorBloom::new(2048, 4, 9);
    for i in 0..200u64 {
        b.insert(i);
    }
    let queries: Vec<u64> = (0..400).collect();
    let batch = b.batch_contains(&queries);
    for (i, &q) in queries.iter().enumerate() {
        assert_eq!(batch[i], b.contains(q), "q={q}");
    }
}

#[test]
fn batch_no_false_negatives() {
    let mut b = TensorBloom::new(4096, 5, 3);
    let members: Vec<u64> = (0..300).map(|i| i * 2).collect();
    for &m in &members {
        b.insert(m);
    }
    let hits = b.batch_contains(&members);
    assert!(hits.iter().all(|&h| h), "a member was reported absent");
}

#[test]
fn removal_deletes_membership() {
    let mut b = TensorBloom::new(8192, 6, 5);
    b.insert(123);
    assert!(b.contains(123));
    b.remove(123);
    // With high probability all of its slots are now empty.
    assert!(!b.contains(123));
}

#[test]
fn fpr_increases_with_load() {
    let mut b = TensorBloom::new(1024, 4, 1);
    let low = b.estimated_fpr();
    for i in 0..400u64 {
        b.insert(i);
    }
    assert!(b.estimated_fpr() > low);
}

#[test]
fn empty_batch_is_empty() {
    let b = TensorBloom::new(256, 3, 1);
    assert!(b.batch_contains(&[]).is_empty());
}
