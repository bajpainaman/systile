//! A Bloom filter whose batch membership test is a single matmul.
//!
//! Run with `cargo run --release --example bloom_membership`.

use systile::prelude::*;

fn main() {
    let mut bloom = TensorBloom::new(2048, 5, 0xB1005);

    // Insert 300 items (the even numbers 0,2,...,598).
    for i in 0..300u64 {
        bloom.insert(i * 2);
    }
    println!("{bloom:?}");

    // Query 0..600 in one matmul of signatures against the filter.
    let queries: Vec<u64> = (0..600).collect();
    let hits = bloom.batch_contains(&queries);

    // Members must never be missed; non-members are rejected up to the FPR.
    let mut false_pos = 0;
    let mut false_neg = 0;
    for (q, &present) in queries.iter().zip(&hits) {
        let truly_in = q % 2 == 0 && *q < 600;
        if truly_in && !present {
            false_neg += 1;
        }
        if !truly_in && present {
            false_pos += 1;
        }
    }

    println!("false negatives: {false_neg} (must be 0)");
    println!(
        "false positives: {false_pos}/{} odd queries (~{:.1}% empirical)",
        300,
        100.0 * false_pos as f64 / 300.0
    );
    assert_eq!(false_neg, 0);
    println!("\n✓ no false negatives, ever — membership decided by one matmul.");
}
