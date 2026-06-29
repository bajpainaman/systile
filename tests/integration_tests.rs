//! End-to-end tests that exercise several features together, the way a real
//! caller would chain them.

use systile::prelude::*;

fn ramp(rows: usize, cols: usize) -> PaddedTileLattice<f32> {
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    PaddedTileLattice::from_dense(rows, cols, &data, Geometry::TPU_V).unwrap()
}

