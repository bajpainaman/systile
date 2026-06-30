//! Levenshtein edit distance as a shortest path over the tropical semiring.
//!
//! Run with `cargo run --release --example edit_distance`.

use systile::prelude::*;

fn main() {
    let ed = TensorEditDistance::new();

    let pairs = [
        ("kitten", "sitting"),
        ("flaw", "lawn"),
        ("gumbo", "gambol"),
        ("", "abc"),
        ("same", "same"),
    ];

    println!("edit distance via tropical (min-plus) matmul:\n");
    for (a, b) in pairs {
        let d = ed.distance_str(a, b);
        println!("  {a:>8} -> {b:<8}  {d}");
    }

    // The classic example.
    assert_eq!(ed.distance_str("kitten", "sitting"), 3);
    assert_eq!(ed.distance_str("same", "same"), 0);
    println!("\n✓ alignment-grid shortest path computed by iterated min-plus matmul.");
}
