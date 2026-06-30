//! Tests for the Count-Min sketch.

use systile::CountMinSketch;

#[test]
fn shape_and_total() {
    let mut cms = CountMinSketch::new(4, 256, 1);
    cms.add(1, 5);
    cms.add(2, 3);
    assert_eq!(cms.rows(), 4);
    assert_eq!(cms.width(), 256);
    assert_eq!(cms.total(), 8);
}

#[test]
fn never_underestimates() {
    let mut cms = CountMinSketch::new(5, 64, 7);
    for i in 0..200u64 {
        cms.add(i, (i % 7) + 1);
    }
    for i in 0..200u64 {
        assert!(
            cms.estimate(i) >= ((i % 7) + 1) as f32,
            "underestimated {i}"
        );
    }
}

#[test]
fn exact_when_no_collisions() {
    // Wide table, few items -> collision-free, estimates exact.
    let mut cms = CountMinSketch::new(4, 4096, 3);
    for i in 0..20u64 {
        cms.add(i, i + 1);
    }
    for i in 0..20u64 {
        assert_eq!(cms.estimate(i), (i + 1) as f32, "item {i}");
    }
}

#[test]
fn batch_matches_scalar() {
    let mut cms = CountMinSketch::new(4, 128, 9);
    for i in 0..100u64 {
        cms.add(i, i % 5 + 1);
    }
    let items: Vec<u64> = (0..100).collect();
    let batch = cms.batch_estimate(&items);
    for (i, &est) in batch.iter().enumerate() {
        assert_eq!(est, cms.estimate(i as u64), "item {i}");
    }
}

#[test]
fn insert_increments_by_one() {
    let mut cms = CountMinSketch::new(3, 1024, 1);
    cms.insert(42);
    cms.insert(42);
    cms.insert(42);
    assert!(cms.estimate(42) >= 3.0);
    assert_eq!(cms.total(), 3);
}

#[test]
fn unseen_item_estimates_low() {
    let mut cms = CountMinSketch::new(4, 4096, 5);
    for i in 0..10u64 {
        cms.add(i, 100);
    }
    // An unseen item in a sparse wide table should estimate 0.
    assert_eq!(cms.estimate(9999), 0.0);
}

#[test]
fn empty_batch_is_empty() {
    let cms = CountMinSketch::new(3, 64, 1);
    assert!(cms.batch_estimate(&[]).is_empty());
}
