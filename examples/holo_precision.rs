//! Compare holographic recall when cleanup runs in f32 vs bf16 — the precision a
//! TPU matrix unit actually uses. bf16 buys ~4x matmul throughput; this measures
//! what it costs in recall.
//!
//! Run with `cargo run --release --example holo_precision`.

use systile::prelude::*;

fn recall(got: &[(usize, f32)], truth: &[(usize, usize)]) -> f64 {
    let correct = truth
        .iter()
        .zip(got)
        .filter(|(&(_, want), &(g, _))| g == want)
        .count();
    100.0 * correct as f64 / truth.len() as f64
}

fn main() {
    let dim = 4096;
    let n_values = 512;

    println!("dim = {dim}, value vocabulary = {n_values}");
    println!(
        "{:>8}  {:>10}  {:>10}",
        "entries", "f32 recall", "bf16 recall"
    );

    for &k in &[50usize, 100, 200, 300, 400] {
        let mut mem = HoloMemory::new(dim, n_values, 0xF00D);
        let pairs: Vec<(usize, usize)> = (0..k).map(|i| (i, (i * 13) % n_values)).collect();
        for &(key, val) in &pairs {
            mem.insert(key, val);
        }
        let probes: Vec<Hyper> = pairs.iter().map(|&(key, _)| mem.probe(key)).collect();

        let f32_hits = mem.values().cleanup_batch(&probes);
        let bf16_hits = mem.values().cleanup_batch_bf16(&probes);

        println!(
            "{k:>8}  {:>9.1}%  {:>9.1}%",
            recall(&f32_hits, &pairs),
            recall(&bf16_hits, &pairs)
        );
    }

    println!("\nbf16 tracks f32 closely under capacity and degrades a little earlier near it.");
}
