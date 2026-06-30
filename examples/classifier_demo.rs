//! Train a hyperdimensional classifier by bundling, classify by matmul.
//!
//! We synthesise `K` class "centroids" (each a random pattern of feature levels),
//! draw noisy training and test samples around them, fit by addition, and report
//! test accuracy — all with no gradients.
//!
//! Run with `cargo run --release --example classifier_demo`.

use systile::prelude::*;

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let (n_features, n_levels, n_classes) = (24, 16, 6);
    let mut clf = HoloClassifier::new(10000, n_features, n_levels, n_classes, 0x5A1AD);

    // Random class centroids in feature-level space.
    let centroids: Vec<Vec<usize>> = (0..n_classes)
        .map(|c| {
            (0..n_features)
                .map(|f| (splitmix64(0xC0 ^ (c as u64) << 8 ^ f as u64) as usize) % n_levels)
                .collect()
        })
        .collect();

    // Draw a noisy sample around a centroid by perturbing ~6 features by ±1 level.
    let sample = |centroid: &[usize], r: u64| -> Vec<usize> {
        let mut s = centroid.to_vec();
        let mut st = r;
        for _ in 0..6 {
            st = splitmix64(st);
            let f = (st as usize) % n_features;
            st = splitmix64(st);
            let delta = if st & 1 == 0 { 1i64 } else { -1 };
            s[f] = (s[f] as i64 + delta).clamp(0, n_levels as i64 - 1) as usize;
        }
        s
    };

    // Train: 40 noisy samples per class.
    for (c, centroid) in centroids.iter().enumerate() {
        for i in 0..40 {
            clf.train(&sample(centroid, 0x7000 + (c as u64) * 1000 + i), c);
        }
    }
    println!("{clf:?}");

    // Test: 50 fresh noisy samples per class, classified in one matmul each batch.
    let mut samples: Vec<Vec<usize>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    for (c, centroid) in centroids.iter().enumerate() {
        for i in 0..50 {
            samples.push(sample(centroid, 0xE000 + (c as u64) * 1000 + i));
            labels.push(c);
        }
    }
    let refs: Vec<&[usize]> = samples.iter().map(|v| v.as_slice()).collect();
    let preds = clf.classify_batch(&refs);

    let correct = preds.iter().zip(&labels).filter(|(p, l)| p == l).count();
    println!(
        "test accuracy: {:.1}% ({correct}/{})",
        100.0 * correct as f64 / labels.len() as f64,
        labels.len()
    );
    println!("\n✓ trained by bundling, classified by matmul — no gradients, no epochs.");
}
