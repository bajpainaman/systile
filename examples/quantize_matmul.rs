//! Quantise to int8, multiply, dequantise, and measure the error against f32.
//!
//! Run with `cargo run --example quantize_matmul`.

use systile::prelude::*;

fn main() {
    let a_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.25).collect();
    let b_data: Vec<f32> = (0..16).map(|i| (i as f32 - 7.0) * 0.5).collect();

    let a = PaddedTileLattice::from_dense(4, 4, &a_data, Geometry::TPU_V).unwrap();
    let b = PaddedTileLattice::from_dense(4, 4, &b_data, Geometry::TPU_V).unwrap();

    // Reference result in f32.
    let reference = a.matmul(&b).unwrap();

    // Calibrate symmetric int8 params from each operand's dynamic range.
    let qa = QuantParams::symmetric(a.abs_max());
    let qb = QuantParams::symmetric(b.abs_max());
    let a_q = a.quantize(qa).unwrap();
    let b_q = b.quantize(qb).unwrap();

    // Dequantise then multiply to approximate an int8 matmul pipeline.
    let approx = a_q
        .dequantize(qa)
        .unwrap()
        .matmul(&b_q.dequantize(qb).unwrap())
        .unwrap();

    let max_err = reference
        .to_dense()
        .iter()
        .zip(approx.to_dense().iter())
        .map(|(r, a)| (r - a).abs())
        .fold(0.0f32, f32::max);

    println!("max abs error from int8 round-trip: {max_err:.4}");
}
