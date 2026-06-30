//! The discrete Fourier transform as a Fourier-matrix matmul.
//!
//! Run with `cargo run --release --example dft_spectrum`.

use std::f32::consts::PI;
use systile::prelude::*;

fn main() {
    let n = 16;
    let dft = TensorDFT::new(n);

    // A signal that is a pure cosine at frequency 3 plus a DC offset of 2.
    let signal: Vec<f32> = (0..n)
        .map(|t| 2.0 + (2.0 * PI * 3.0 * t as f32 / n as f32).cos())
        .collect();

    let mag = dft.magnitude(&signal);
    println!("magnitude spectrum (matmul by the Fourier matrix):");
    for (k, m) in mag.iter().enumerate() {
        let bar = "#".repeat((m / 2.0) as usize);
        println!("  bin {k:>2}: {m:>6.1} {bar}");
    }

    // DC bin = N * mean = sum of the signal; frequency-3 energy shows at bins 3 & 13.
    let dc = mag[0];
    println!(
        "\nDC bin magnitude {dc:.1} (≈ sum = {:.1})",
        signal.iter().sum::<f32>()
    );

    // Round-trip: inverse(forward(x)) ≈ x.
    let (re, im) = dft.forward(&signal);
    let recon = dft.inverse(&re, &im);
    let max_err = signal
        .iter()
        .zip(&recon)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("inverse round-trip max error: {max_err:.2e}");
    assert!(max_err < 1e-3);
    println!("✓ DFT and its inverse computed as dense matmuls.");
}
