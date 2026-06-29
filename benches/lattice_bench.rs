//! A dependency-free micro-benchmark harness for the hot lattice operations.
//!
//! This uses `harness = false` and a hand-rolled timing loop so the crate stays
//! free of dev-dependencies. Run with `cargo bench`.

use std::time::Instant;
use systile::prelude::*;

fn bench<F: FnMut()>(name: &str, iters: u32, mut f: F) {
    // Warm up so the first allocation does not dominate the measurement.
    for _ in 0..3 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per = elapsed.as_secs_f64() / iters as f64 * 1e6;
    println!("{name:<28} {per:>10.3} us/iter  ({iters} iters)");
}

fn make_square(n: usize) -> PaddedTileLattice<f32> {
    let data: Vec<f32> = (0..n * n).map(|i| (i % 17) as f32).collect();
    PaddedTileLattice::from_dense(n, n, &data, Geometry::TPU_V).unwrap()
}

fn main() {
    println!("systile lattice benchmarks\n");

    let a = make_square(128);
    let b = make_square(128);
    bench("matmul 128x128", 20, || {
        let _ = a.matmul(&b).unwrap();
    });

    bench("transpose 128x128", 200, || {
        let _ = a.transpose();
    });

    bench("to_dense 128x128", 500, || {
        let _ = a.to_dense();
    });

    bench("tile_sparsity 128x128", 500, || {
        let _ = a.tile_sparsity();
    });

    let q = QuantParams::symmetric(a.abs_max());
    bench("quantize 128x128", 200, || {
        let _ = a.quantize(q).unwrap();
    });
}
