//! Tests for the hyperdimensional classifier.

use systile::HoloClassifier;

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn shape_accessors() {
    let clf = HoloClassifier::new(2048, 8, 10, 4, 1);
    assert_eq!(clf.n_classes(), 4);
    assert_eq!(clf.class_count(0), 0);
}

#[test]
fn training_increments_counts() {
    let mut clf = HoloClassifier::new(2048, 4, 8, 3, 1);
    clf.train(&[0, 1, 2, 3], 1);
    clf.train(&[1, 1, 1, 1], 1);
    assert_eq!(clf.class_count(1), 2);
    assert_eq!(clf.class_count(0), 0);
}

#[test]
fn memorises_distinct_prototypes() {
    // One sample per class, all very different -> each classifies to itself.
    let mut clf = HoloClassifier::new(8192, 10, 12, 3, 7);
    let a = [0usize, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let b = [11usize, 11, 11, 11, 11, 11, 11, 11, 11, 11];
    let c = [5usize, 6, 5, 6, 5, 6, 5, 6, 5, 6];
    clf.train(&a, 0);
    clf.train(&b, 1);
    clf.train(&c, 2);
    assert_eq!(clf.classify(&a), 0);
    assert_eq!(clf.classify(&b), 1);
    assert_eq!(clf.classify(&c), 2);
}

#[test]
fn encode_is_deterministic() {
    let clf = HoloClassifier::new(1024, 4, 8, 2, 3);
    assert_eq!(clf.encode(&[1, 2, 3, 4]), clf.encode(&[1, 2, 3, 4]));
}

#[test]
fn classifies_noisy_clusters_accurately() {
    let (n_features, n_levels, n_classes) = (20, 16, 5);
    let mut clf = HoloClassifier::new(10000, n_features, n_levels, n_classes, 0xC1A);

    let centroids: Vec<Vec<usize>> = (0..n_classes)
        .map(|c| {
            (0..n_features)
                .map(|f| (splitmix64(0xA0 ^ (c as u64) << 8 ^ f as u64) as usize) % n_levels)
                .collect()
        })
        .collect();

    let sample = |centroid: &[usize], r: u64| -> Vec<usize> {
        let mut s = centroid.to_vec();
        let mut st = r;
        for _ in 0..5 {
            st = splitmix64(st);
            let f = (st as usize) % n_features;
            st = splitmix64(st);
            let d = if st & 1 == 0 { 1i64 } else { -1 };
            s[f] = (s[f] as i64 + d).clamp(0, n_levels as i64 - 1) as usize;
        }
        s
    };

    for (c, centroid) in centroids.iter().enumerate() {
        for i in 0..30 {
            clf.train(&sample(centroid, 0x1000 + (c as u64) * 100 + i), c);
        }
    }

    let mut samples = Vec::new();
    let mut labels = Vec::new();
    for (c, centroid) in centroids.iter().enumerate() {
        for i in 0..30 {
            samples.push(sample(centroid, 0x9000 + (c as u64) * 100 + i));
            labels.push(c);
        }
    }
    let refs: Vec<&[usize]> = samples.iter().map(|v| v.as_slice()).collect();
    let preds = clf.classify_batch(&refs);
    let correct = preds.iter().zip(&labels).filter(|(p, l)| p == l).count();
    assert!(
        correct as f64 / labels.len() as f64 > 0.9,
        "accuracy {correct}/{} below 90%",
        labels.len()
    );
}

#[test]
fn empty_batch_is_empty() {
    let clf = HoloClassifier::new(1024, 4, 8, 2, 1);
    assert!(clf.classify_batch(&[]).is_empty());
}
