//! Build a block-sparse lattice and report which tiles a kernel can skip.
//!
//! Run with `cargo run --example sparsity_report`.

use systile::prelude::*;

fn main() {
    // 256x256 with only the diagonal tiles populated.
    let n = 256;
    let geom = Geometry::TPU_V;
    let mut lattice = PaddedTileLattice::<f32>::zeroed(n, n, geom).unwrap();
    for i in 0..n {
        lattice.set(i, i, 1.0).unwrap();
    }

    println!("lattice: {}x{}", lattice.rows(), lattice.cols());
    println!("tiles: {}", lattice.num_tiles());
    println!("zero tiles: {}", lattice.count_zero_tiles());
    println!("tile sparsity: {:.1}%", lattice.tile_sparsity() * 100.0);

    let live = lattice.nonzero_tile_coords();
    println!("\n{} tiles must be fed through the array:", live.len());
    for (r, c) in live.iter().take(8) {
        println!("  tile ({r}, {c})");
    }
}
