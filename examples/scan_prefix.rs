//! Prefix sums as a triangular matmul.
//!
//! Run with `cargo run --release --example scan_prefix`.

use systile::prelude::*;

fn main() {
    let scan = TensorScan::new();
    let x = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];

    println!("input:      {x:?}");
    println!("inclusive:  {:?}", scan.inclusive(&x));
    println!("exclusive:  {:?}", scan.exclusive(&x));
    println!("suffix:     {:?}", scan.suffix(&x));
    println!("total:      {}", scan.total(&x));

    // Cross-check the inclusive scan against a sequential cumsum.
    let mut acc = 0.0;
    let expected: Vec<f32> = x
        .iter()
        .map(|&v| {
            acc += v;
            acc
        })
        .collect();
    assert_eq!(scan.inclusive(&x), expected);
    println!("\n✓ cumulative sum computed as L·x — O(1) depth, no carried loop.");
}
