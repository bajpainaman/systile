//! Exact nearest-neighbour search as a matmul: index random vectors, plant a known
//! near-duplicate, and recover it as the top hit.
//!
//! Run with `cargo run --release --example index_search`.

use systile::prelude::*;

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let h = splitmix64(seed.wrapping_add(i as u64));
            (h as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn main() {
    let dim = 256;
    let n = 2000;
    let mut index = TensorIndex::new(dim);
    for j in 0..n {
        index.add(random_vec(dim, 0x100 + j as u64));
    }
    println!("{index:?}");

    // Pick item 1234 and make a noisy near-duplicate query.
    let target = 1234usize;
    let mut query = index.vector(target).to_vec();
    for (i, x) in query.iter_mut().enumerate() {
        if i % 7 == 0 {
            *x += 0.05; // small perturbation
        }
    }

    // One (1 x 256) . (256 x 2000) matmul scores the whole corpus.
    let hits = index.search(&query, 5);
    println!("\ntop-5 nearest to a noisy copy of item {target}:");
    for (rank, h) in hits.iter().enumerate() {
        println!("  #{}: item {} (score {:.2})", rank + 1, h.id, h.score);
    }
    assert_eq!(hits[0].id, target);
    println!("\n✓ exact nearest neighbour recovered by a single dense matmul.");
}
