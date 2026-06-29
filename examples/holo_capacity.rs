//! Measure how recall degrades as you overfill a holographic store, and compare
//! the empirical cliff to the `K_max ≈ d / (2 ln M)` capacity rule of thumb.
//!
//! Run with `cargo run --release --example holo_capacity`.

use systile::prelude::*;

fn recall_at(dim: usize, n_values: usize, n_pairs: usize, seed: u64) -> f64 {
    let mut mem = HoloMemory::new(dim, n_values, seed);
    let pairs: Vec<(usize, usize)> = (0..n_pairs)
        .map(|i| (i, (i * 2654435761) % n_values))
        .collect();
    for &(k, v) in &pairs {
        mem.insert(k, v);
    }
    let keys: Vec<usize> = pairs.iter().map(|&(k, _)| k).collect();
    let got = mem.batch_get(&keys);
    let correct = pairs
        .iter()
        .zip(&got)
        .filter(|(&(_, want), &(g, _))| g == want)
        .count();
    correct as f64 / n_pairs as f64
}

fn main() {
    let dim = 4096;
    let n_values = 512;
    let predicted = dim as f64 / (2.0 * (n_values as f64).ln());

    println!("dim = {dim}, value vocabulary M = {n_values}");
    println!("predicted capacity K_max ≈ d / (2 ln M) = {predicted:.0}\n");
    println!("{:>8}  {:>8}  {:>10}", "entries", "recall", "load");

    for &k in &[50usize, 100, 200, 300, 400, 500, 700, 1000] {
        let recall = recall_at(dim, n_values, k, 0x5EED);
        let load = k as f64 / predicted;
        let bar = "#".repeat((recall * 20.0) as usize);
        println!("{k:>8}  {:>7.1}%  {load:>9.2}  {bar}", recall * 100.0);
    }

    println!("\nRecall stays near-perfect below the predicted capacity and falls off above it.");
}
