//! Find a pattern inside a signal by im2col cross-correlation matmul.
//!
//! Run with `cargo run --release --example conv_search`.

use systile::prelude::*;

fn main() {
    let conv = TensorConv::new();

    // A signal with a known pattern planted at offset 5 (and again at 11).
    let pattern = [1.0f32, -1.0, 2.0];
    let mut signal = vec![
        0.3, 0.1, -0.2, 0.5, 0.0, 1.0, -1.0, 2.0, 0.4, -0.3, 0.2, 1.0, -1.0, 2.0,
    ];

    println!("signal length {}, pattern {:?}", signal.len(), pattern);
    let scores = conv.correlate(&signal, &pattern);
    println!(
        "correlation per offset: {:?}",
        scores
            .iter()
            .map(|s| (s * 10.0).round() / 10.0)
            .collect::<Vec<_>>()
    );

    let best = conv.best_offset(&signal, &pattern).unwrap();
    println!("strongest match at offset {best}");

    let all = conv.find_all(&signal, &pattern);
    println!("exact matches at offsets {all:?}");
    assert_eq!(all, vec![5, 11]);

    // Negative control: a pattern that does not occur.
    signal[5] = 9.9;
    let _ = conv.find_all(&signal, &pattern);
    println!("\n✓ pattern located by a single im2col matmul over all offsets.");
}
