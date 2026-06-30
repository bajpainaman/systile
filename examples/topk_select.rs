//! Select the top-k by a comparison-count matmul.
//!
//! Run with `cargo run --release --example topk_select`.

use systile::prelude::*;

fn main() {
    let topk = TensorTopK::new();
    let scores = [0.2f32, 0.9, 0.5, 0.95, 0.1, 0.7, 0.3, 0.85];

    println!("scores: {scores:?}");
    println!("ranks (C·1): {:?}", topk.ranks(&scores));

    let top3 = topk.select(&scores, 3);
    println!("\ntop-3 (index, score), best first:");
    for (rank, (idx, score)) in top3.iter().enumerate() {
        println!("  #{}: index {idx} = {score}", rank + 1);
    }

    assert_eq!(topk.select_indices(&scores, 3), vec![3, 1, 7]);
    println!("\n3rd largest value: {:?}", topk.kth_largest(&scores, 3));
    println!("✓ top-k chosen by one comparison matmul, no full sort.");
}
