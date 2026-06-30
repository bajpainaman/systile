//! Capstone: compose pillars into one working system — retrieval-augmented
//! classification, end to end in matmuls.
//!
//! Pipeline per query:
//!   1. `TensorIndex` (pillar 5) finds the nearest corpus items — one matmul.
//!   2. `TensorAttention` (pillar 16) softly pools the neighbours' one-hot class
//!      vectors, weighted by similarity to the query — three matmuls.
//!   3. argmax of the pooled distribution is the predicted class.
//!
//! No training: the "model" is just the indexed corpus. Run with
//! `cargo run --release --example capstone_rag`.

use systile::prelude::*;

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn noisy(centroid: &[f32], seed: u64, scale: f32) -> Vec<f32> {
    centroid
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let h = splitmix64(seed.wrapping_add(i as u64));
            c + ((h as f32 / u64::MAX as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn main() {
    let (dim, n_classes, per_class) = (128, 5, 40);
    let topk = TensorTopK::new();
    let attn = TensorAttention::new(dim);

    // Class centroids.
    let centroids: Vec<Vec<f32>> = (0..n_classes)
        .map(|c| noisy(&vec![0.0; dim], 0xC000 + c as u64, 1.0))
        .collect();

    // Build the corpus index and remember each item's class as a one-hot vector.
    let mut index = TensorIndex::new(dim);
    let mut labels: Vec<Vec<f32>> = Vec::new();
    for c in 0..n_classes {
        for i in 0..per_class {
            index.add(noisy(&centroids[c], 0x1000 + (c as u64) * 1000 + i, 0.6));
            let mut onehot = vec![0.0f32; n_classes];
            onehot[c] = 1.0;
            labels.push(onehot);
        }
    }
    println!(
        "corpus: {} items, {n_classes} classes, dim {dim}",
        index.len()
    );

    // Evaluate the retrieve→attend→decide pipeline on fresh queries.
    let k = 8;
    let mut correct = 0;
    let mut total = 0;
    for c in 0..n_classes {
        for i in 0..30 {
            let query = noisy(&centroids[c], 0x9000 + (c as u64) * 1000 + i, 0.6);

            // 1. retrieve nearest neighbours (one matmul inside).
            let hits = index.search(&query, k);
            // (top-k by score is already sorted; TensorTopK would re-rank a raw row)
            let _ = topk.ranks(&hits.iter().map(|h| h.score).collect::<Vec<_>>());

            // 2. attention over the neighbours' vectors (keys) and class one-hots (values).
            let keys: Vec<&[f32]> = hits.iter().map(|h| index.vector(h.id)).collect();
            let vals: Vec<&[f32]> = hits.iter().map(|h| labels[h.id].as_slice()).collect();
            let dist = attn.attend_one(&query, &keys, &vals);

            // 3. decide.
            let pred = (0..n_classes)
                .max_by(|&a, &b| dist[a].partial_cmp(&dist[b]).unwrap())
                .unwrap();
            if pred == c {
                correct += 1;
            }
            total += 1;
        }
    }

    println!(
        "retrieval-augmented accuracy: {:.1}% ({correct}/{total})",
        100.0 * correct as f64 / total as f64
    );
    assert!(correct as f64 / total as f64 > 0.9);
    println!("\n✓ index → attention → decision — pillars composed, all matmuls.");
}
