//! Estimate item frequencies with a Count-Min sketch whose batch query is a matmul.
//!
//! Run with `cargo run --release --example sketch_frequency`.

use systile::prelude::*;

fn main() {
    let mut cms = CountMinSketch::new(4, 512, 0x5037CB);

    // A skewed stream: item i appears (i+1) times, for i in 0..20.
    for i in 0..20u64 {
        cms.add(i, i + 1);
    }
    println!("{cms:?}");

    // Batch-estimate all 20 items in d matmuls (one per hash row).
    let items: Vec<u64> = (0..20).collect();
    let estimates = cms.batch_estimate(&items);

    let mut max_over = 0.0f32;
    println!("{:>4} {:>6} {:>9}", "item", "true", "estimate");
    for (i, &est) in estimates.iter().enumerate() {
        let truth = (i + 1) as f32;
        max_over = max_over.max(est - truth);
        if !(6..=16).contains(&i) {
            println!("{i:>4} {truth:>6} {est:>9}");
        }
        assert!(
            est >= truth,
            "Count-Min must never underestimate (item {i})"
        );
    }
    println!("\nmax overestimate across all items: {max_over}");
    println!("✓ frequencies estimated by matmul; never an underestimate.");
}
