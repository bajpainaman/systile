//! Tests for the matmul discrete Fourier transform.

use std::f32::consts::PI;
use systile::TensorDFT;

#[test]
fn dc_bin_is_sum() {
    let dft = TensorDFT::new(8);
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (re, im) = dft.forward(&x);
    assert!((re[0] - x.iter().sum::<f32>()).abs() < 1e-3);
    assert!(im[0].abs() < 1e-3);
}

#[test]
fn round_trip_reconstructs_signal() {
    let dft = TensorDFT::new(16);
    let x: Vec<f32> = (0..16)
        .map(|i| (i as f32 * 0.7).sin() + 0.3 * i as f32)
        .collect();
    let (re, im) = dft.forward(&x);
    let recon = dft.inverse(&re, &im);
    for (a, b) in x.iter().zip(&recon) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}

#[test]
fn pure_cosine_peaks_at_its_frequency() {
    let n = 32;
    let dft = TensorDFT::new(n);
    let freq = 5;
    let x: Vec<f32> = (0..n)
        .map(|t| (2.0 * PI * freq as f32 * t as f32 / n as f32).cos())
        .collect();
    let mag = dft.magnitude(&x);
    // Energy concentrates at bins `freq` and `n - freq`.
    let peak = (0..n)
        .max_by(|&a, &b| mag[a].partial_cmp(&mag[b]).unwrap())
        .unwrap();
    assert!(peak == freq || peak == n - freq, "peak at bin {peak}");
}

#[test]
fn magnitude_is_symmetric_for_real_input() {
    let n = 16;
    let dft = TensorDFT::new(n);
    let x: Vec<f32> = (0..n).map(|i| (i as f32).cos()).collect();
    let mag = dft.magnitude(&x);
    for k in 1..n / 2 {
        assert!((mag[k] - mag[n - k]).abs() < 1e-2, "bin {k}");
    }
}

#[test]
fn impulse_has_flat_spectrum() {
    let n = 8;
    let dft = TensorDFT::new(n);
    let mut x = vec![0.0f32; n];
    x[0] = 1.0;
    let mag = dft.magnitude(&x);
    for m in &mag {
        assert!((m - 1.0).abs() < 1e-3);
    }
}

#[test]
fn complex_forward_matches_real_for_zero_imag() {
    let dft = TensorDFT::new(8);
    let x = [1.0f32, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
    let (r1, i1) = dft.forward(&x);
    let (r2, i2) = dft.forward_complex(&x, &[0.0; 8]);
    for k in 0..8 {
        assert!((r1[k] - r2[k]).abs() < 1e-4 && (i1[k] - i2[k]).abs() < 1e-4);
    }
}
