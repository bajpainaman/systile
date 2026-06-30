//! Tests for the holographic set.

use systile::HoloSet;

#[test]
fn empty_set_contains_nothing() {
    let s = HoloSet::new(2048, 64, 1);
    assert!(s.is_empty());
    assert!(!s.contains(0));
}

#[test]
fn inserted_member_is_contained() {
    let mut s = HoloSet::new(4096, 64, 1);
    s.insert(7);
    assert!(s.contains(7));
}

#[test]
fn non_member_is_not_contained() {
    let mut s = HoloSet::new(8192, 256, 2);
    for id in [1usize, 5, 9, 13] {
        s.insert(id);
    }
    assert!(!s.contains(100));
}

#[test]
fn all_members_contained() {
    let mut s = HoloSet::new(8192, 256, 3);
    let members = [2usize, 4, 8, 16, 32, 64];
    for &m in &members {
        s.insert(m);
    }
    for &m in &members {
        assert!(s.contains(m), "missing {m}");
    }
}

#[test]
fn member_similarity_exceeds_nonmember() {
    let mut s = HoloSet::new(8192, 256, 4);
    s.insert(10);
    s.insert(20);
    assert!(s.similarity(10) > 0.5);
    assert!(s.similarity(200) < 0.5);
}

#[test]
fn batch_contains_matches_contains() {
    let mut s = HoloSet::new(8192, 128, 5);
    for id in [1usize, 3, 5, 7, 9] {
        s.insert(id);
    }
    let probe: Vec<usize> = (0..20).collect();
    let batch = s.batch_contains(&probe);
    for (i, &id) in probe.iter().enumerate() {
        assert_eq!(batch[i], s.contains(id), "id {id}");
    }
}

#[test]
fn estimated_cardinality_tracks_inserts() {
    let mut s = HoloSet::new(8192, 256, 6);
    for id in 0..10 {
        s.insert(id);
    }
    // Squared-norm estimator should be close to the true count.
    let est = s.estimated_cardinality();
    assert!((est - 10.0).abs() < 3.0, "estimate was {est}");
}

#[test]
fn union_contains_both_sides() {
    let mut a = HoloSet::new(8192, 256, 7);
    let mut b = HoloSet::new(8192, 256, 7);
    a.insert(1);
    a.insert(2);
    b.insert(100);
    b.insert(101);
    let u = a.union(&b);
    assert!(u.contains(1));
    assert!(u.contains(2));
    assert!(u.contains(100));
    assert!(u.contains(101));
}

#[test]
fn union_length_adds() {
    let mut a = HoloSet::new(4096, 64, 8);
    let mut b = HoloSet::new(4096, 64, 8);
    a.insert(0);
    b.insert(1);
    b.insert(2);
    assert_eq!(a.union(&b).len(), 3);
}
