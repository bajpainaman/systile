//! Inspect how an awkward shape pads up to tile boundaries.
//!
//! Run with `cargo run --example padding_inspect`.

use systile::prelude::*;

fn main() {
    for &(rows, cols) in &[(1usize, 1usize), (3, 5), (8, 128), (130, 257)] {
        let shape = Shape::new(rows, cols, &Geometry::TPU_V);
        println!(
            "{rows:>4}x{cols:<4} -> {:>4}x{:<4} padded  |  {:>6.1}% padding",
            shape.padded_rows,
            shape.padded_cols,
            shape.padding_ratio() * 100.0
        );
    }
}
