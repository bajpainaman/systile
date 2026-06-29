//! Tests for the holographic key→value store.

use systile::HoloMemory;

#[test]
fn empty_memory_reports_empty() {
    let m = HoloMemory::new(1024, 16, 1);
    assert!(m.is_empty());
    assert_eq!(m.len(), 0);
}

#[test]
fn single_insert_is_recovered() {
    let mut m = HoloMemory::new(4096, 64, 1);
    m.insert(0, 42);
    assert_eq!(m.get(0).0, 42);
}

#[test]
fn len_counts_inserts() {
    let mut m = HoloMemory::new(2048, 32, 1);
    m.insert(0, 1);
    m.insert(1, 2);
    assert_eq!(m.len(), 2);
}

#[test]
fn distinct_keys_map_to_distinct_values() {
    let mut m = HoloMemory::new(8192, 100, 7);
    for k in 0..20 {
        m.insert(k, (k * 3) % 100);
    }
    for k in 0..20 {
        assert_eq!(m.get(k).0, (k * 3) % 100, "key {k}");
    }
}

#[test]
fn under_capacity_recall_is_perfect() {
    let mut m = HoloMemory::new(8192, 256, 123);
    let pairs: Vec<(usize, usize)> = (0..50).map(|i| (i, (i * 11) % 256)).collect();
    for &(k, v) in &pairs {
        m.insert(k, v);
    }
    let got = m.batch_get(&pairs.iter().map(|&(k, _)| k).collect::<Vec<_>>());
    let correct = pairs
        .iter()
        .zip(&got)
        .filter(|(&(_, want), &(g, _))| g == want)
        .count();
    assert_eq!(
        correct,
        pairs.len(),
        "expected perfect recall under capacity"
    );
}

#[test]
fn batch_get_matches_get() {
    let mut m = HoloMemory::new(4096, 64, 9);
    for k in 0..30 {
        m.insert(k, (k * 5) % 64);
    }
    let keys: Vec<usize> = (0..30).collect();
    let batch = m.batch_get(&keys);
    for (k, (bv, _)) in keys.iter().zip(&batch) {
        assert_eq!(m.get(*k).0, *bv);
    }
}

#[test]
fn remove_cancels_insert() {
    let mut m = HoloMemory::new(4096, 64, 3);
    m.insert(0, 10);
    m.insert(1, 20);
    let before = m.get(1).0;
    m.insert(0, 10);
    m.remove(0, 10);
    // Removing the duplicate leaves key 1 intact.
    assert_eq!(m.get(1).0, before);
}

#[test]
fn capacity_grows_with_dim() {
    let small = HoloMemory::new(1024, 64, 1);
    let big = HoloMemory::new(8192, 64, 1);
    assert!(big.estimated_capacity() > small.estimated_capacity());
}

#[test]
fn load_factor_tracks_inserts() {
    let mut m = HoloMemory::new(4096, 64, 1);
    let cap = m.estimated_capacity();
    for k in 0..10 {
        m.insert(k, k);
    }
    assert!((m.load_factor() - 10.0 / cap).abs() < 1e-9);
}

#[test]
fn probe_dimension_matches() {
    let mut m = HoloMemory::new(2048, 32, 1);
    m.insert(3, 7);
    assert_eq!(m.probe(3).dim(), 2048);
}

#[test]
fn overcapacity_recall_degrades() {
    // Far past capacity, recall should be well below perfect.
    let mut m = HoloMemory::new(1024, 256, 5);
    let pairs: Vec<(usize, usize)> = (0..800).map(|i| (i, (i * 7) % 256)).collect();
    for &(k, v) in &pairs {
        m.insert(k, v);
    }
    let got = m.batch_get(&pairs.iter().map(|&(k, _)| k).collect::<Vec<_>>());
    let correct = pairs
        .iter()
        .zip(&got)
        .filter(|(&(_, want), &(g, _))| g == want)
        .count();
    assert!(
        correct < pairs.len(),
        "expected some recall failures past capacity"
    );
}
