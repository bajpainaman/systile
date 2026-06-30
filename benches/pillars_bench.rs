//! Dependency-free micro-benchmarks for the headline operation of each pillar.
//!
//! Uses `harness = false` and a hand-rolled timing loop so the crate stays free of
//! dev-dependencies. Run with `cargo bench --bench pillars_bench`.

use std::time::Instant;
use systile::prelude::*;

fn bench<F: FnMut()>(name: &str, iters: u32, mut f: F) {
    for _ in 0..2 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let per = start.elapsed().as_secs_f64() / iters as f64 * 1e6;
    println!("{name:<34} {per:>11.2} us/iter  ({iters} iters)");
}

fn main() {
    println!("systile pillar benchmarks\n");

    // Pillar 1: holographic key->value batch lookup (one big matmul).
    let mut mem = HoloMemory::new(8192, 1000, 1);
    for k in 0..200 {
        mem.insert(k, (k * 7) % 1000);
    }
    let keys: Vec<usize> = (0..200).collect();
    bench("holo batch_get 200x8192x1000", 20, || {
        let _ = mem.batch_get(&keys);
    });

    // Pillar 1: resonator factorization (iterated matmul cleanup).
    let res = Resonator::uniform(4096, 3, 40, 7);
    let composite = res.compose(&[3, 17, 29]);
    bench("resonator factorize 40^3", 5, || {
        let _ = res.factorize(&composite, 200);
    });

    // Pillar 2: all-pairs shortest paths (tropical matrix powers).
    let mut g = TensorGraph::new(64);
    for i in 0..64 {
        g.add_edge(i, (i + 1) % 64, 1.0);
        g.add_edge(i, (i + 7) % 64, 2.0);
    }
    bench("graph shortest_paths n=64", 20, || {
        let _ = g.shortest_paths();
    });

    // Pillar 5: exact k-NN search (one matmul over the corpus).
    let mut index = TensorIndex::new(256);
    for j in 0..2000 {
        index.add((0..256).map(|i| ((i * 31 + j) % 17) as f32).collect());
    }
    let q: Vec<f32> = index.vector(1234).to_vec();
    bench("index search 1x256 . 256x2000", 50, || {
        let _ = index.search(&q, 10);
    });

    // Pillar 7: sort via comparison matmul.
    let xs: Vec<f32> = (0..128).map(|i| ((i * 37) % 101) as f32).collect();
    let sorter = TensorSort::new();
    bench("sort 128 (comparison matmul)", 100, || {
        let _ = sorter.sort(&xs);
    });

    // Pillar 14: DFT (Fourier-matrix matmul).
    let dft = TensorDFT::new(256);
    let sig: Vec<f32> = (0..256).map(|t| (t as f32 * 0.1).sin()).collect();
    bench("dft magnitude n=256", 100, || {
        let _ = dft.magnitude(&sig);
    });

    // Pillar 16: attention (softmax(QK^T)V).
    let attn = TensorAttention::new(64);
    let kv: Vec<Vec<f32>> = (0..128)
        .map(|j| (0..64).map(|i| ((i + j) % 9) as f32).collect())
        .collect();
    let kref: Vec<&[f32]> = kv.iter().map(|v| v.as_slice()).collect();
    let queries: Vec<Vec<f32>> = (0..32)
        .map(|j| (0..64).map(|i| ((i * 2 + j) % 7) as f32).collect())
        .collect();
    let qref: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();
    bench("attention 32x128 d=64", 50, || {
        let _ = attn.attend(&qref, &kref, &kref);
    });
}
