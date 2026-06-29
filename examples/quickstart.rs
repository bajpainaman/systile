//! The thirty-second tour: build two lattices and multiply them.
//!
//! Run with `cargo run --example quickstart`.

use systile::prelude::*;

fn main() {
    let a =
        PaddedTileLattice::from_dense(2, 3, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], Geometry::TPU_V)
            .unwrap();

    let b =
        PaddedTileLattice::from_dense(3, 2, &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], Geometry::TPU_V)
            .unwrap();

    println!("A is {}x{}", a.rows(), a.cols());
    println!("  stored as {} padded elements", a.padded_len());
    println!("  in {} tile(s)", a.num_tiles());

    let (c, stats) = a.matmul_with_stats(&b).unwrap();
    println!("\nA @ B = {:?}", c.to_dense());
    println!("did {} useful MACs", stats.macs);
    println!("array utilisation: {:.1}%", stats.utilisation() * 100.0);
}
