//! `TensorDFT` — the discrete Fourier transform as a matmul.
//!
//! The DFT is exactly a matrix multiply by the Fourier matrix
//! `F[k,m] = e^{−2πi·km/N}`. The `O(N log N)` FFT is a clever factorisation of that
//! matmul; on hardware where the dense `O(N²)` matmul is cheap, you can just *do*
//! the matmul. `systile` stores `F` as its real and imaginary parts (no complex
//! type needed) and computes the transform on the systolic engine.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;
use core::f32::consts::PI;

/// A discrete Fourier transform of a fixed size `n`, realised as a Fourier-matrix
/// matmul.
#[derive(Clone)]
pub struct TensorDFT {
    n: usize,
    geom: Geometry,
    /// Real part of the Fourier matrix, `cos(2π km / n)`.
    fr: PaddedTileLattice<f32>,
    /// Imaginary part of the Fourier matrix, `−sin(2π km / n)`.
    fi: PaddedTileLattice<f32>,
}

impl TensorDFT {
    /// Build the size-`n` transform.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "transform size must be positive");
        let geom = Geometry::TPU_V;
        let mut fr = vec![0.0f32; n * n];
        let mut fi = vec![0.0f32; n * n];
        for k in 0..n {
            for m in 0..n {
                let theta = 2.0 * PI * (k as f32) * (m as f32) / n as f32;
                fr[k * n + m] = theta.cos();
                fi[k * n + m] = -theta.sin();
            }
        }
        TensorDFT {
            n,
            geom,
            fr: PaddedTileLattice::from_dense(n, n, &fr, geom).unwrap(),
            fi: PaddedTileLattice::from_dense(n, n, &fi, geom).unwrap(),
        }
    }

    /// Transform size.
    #[inline]
    pub fn size(&self) -> usize {
        self.n
    }

    fn matvec(&self, mat: &PaddedTileLattice<f32>, x: &[f32]) -> Vec<f32> {
        let xv = PaddedTileLattice::from_dense(self.n, 1, x, self.geom).unwrap();
        mat.matmul(&xv).unwrap().to_dense()
    }

    /// Forward DFT of a real signal, returning `(real, imag)` spectra.
    pub fn forward(&self, real: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(
            real.len(),
            self.n,
            "signal length must match transform size"
        );
        (self.matvec(&self.fr, real), self.matvec(&self.fi, real))
    }

    /// Forward DFT of a complex signal `(real, imag)`.
    pub fn forward_complex(&self, real: &[f32], imag: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let re = sub(&self.matvec(&self.fr, real), &self.matvec(&self.fi, imag));
        let im = add(&self.matvec(&self.fr, imag), &self.matvec(&self.fi, real));
        (re, im)
    }

    /// Inverse DFT of a spectrum `(real, imag)`, returning the real part of the
    /// reconstructed signal.
    pub fn inverse(&self, real: &[f32], imag: &[f32]) -> Vec<f32> {
        // IDFT uses the conjugate Fourier matrix (imag negated) and a 1/n scale.
        let xr = self.matvec(&self.fr, real);
        let xi_term = self.matvec(&self.fi, imag); // (−sin) applied to imag
                                                   // real(out) = (Fr·re − (−Fi)·im) / n = (Fr·re + Fi·im) / n
        add(&xr, &xi_term)
            .into_iter()
            .map(|v| v / self.n as f32)
            .collect()
    }

    /// The magnitude spectrum `|X[k]|` of a real signal.
    pub fn magnitude(&self, real: &[f32]) -> Vec<f32> {
        let (re, im) = self.forward(real);
        re.iter()
            .zip(&im)
            .map(|(r, i)| (r * r + i * i).sqrt())
            .collect()
    }
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

impl core::fmt::Debug for TensorDFT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "TensorDFT {{ n: {} }}", self.n)
    }
}
