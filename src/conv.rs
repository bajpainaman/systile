//! `TensorConv` — 1-D pattern search as an im2col cross-correlation matmul.
//!
//! Sliding a kernel over a signal and taking a dot product at every offset is a
//! convolution — the workload tensor accelerators were *built* for. The matmul
//! form is the classic **im2col** lowering: gather every length-`k` window of the
//! signal as a row of a matrix `W`, and the correlation at all offsets is one
//! product `W · kernel`.
//!
//! This turns "where does this pattern occur?" into a matmul plus an argmax,
//! exactly the shape an MXU wants.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;

/// A 1-D cross-correlation / pattern-search engine.
#[derive(Clone, Copy, Debug)]
pub struct TensorConv {
    geom: Geometry,
}

impl Default for TensorConv {
    fn default() -> Self {
        TensorConv {
            geom: Geometry::TPU_V,
        }
    }
}

impl TensorConv {
    /// Create a convolution engine with the default tile geometry.
    pub fn new() -> Self {
        TensorConv::default()
    }

    /// Valid cross-correlation of `signal` with `kernel`: a score per offset,
    /// computed as the im2col matmul `W · kernel`. The result has length
    /// `signal.len() − kernel.len() + 1`.
    pub fn correlate(&self, signal: &[f32], kernel: &[f32]) -> Vec<f32> {
        let k = kernel.len();
        assert!(k > 0, "kernel must be non-empty");
        if signal.len() < k {
            return Vec::new();
        }
        let windows = signal.len() - k + 1;
        // Build the im2col matrix: row w is signal[w .. w+k].
        let mut cols = vec![0.0f32; windows * k];
        for (w, row) in cols.chunks_exact_mut(k).enumerate() {
            row.copy_from_slice(&signal[w..w + k]);
        }
        let wlat = PaddedTileLattice::from_dense(windows, k, &cols, self.geom).unwrap();
        let kvec = PaddedTileLattice::from_dense(k, 1, kernel, self.geom).unwrap();
        wlat.matmul(&kvec).unwrap().to_dense()
    }

    /// The offset of the strongest correlation between `signal` and `pattern`, or
    /// `None` if the pattern is longer than the signal.
    pub fn best_offset(&self, signal: &[f32], pattern: &[f32]) -> Option<usize> {
        let scores = self.correlate(signal, pattern);
        if scores.is_empty() {
            return None;
        }
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, &s) in scores.iter().enumerate() {
            if s > best.1 {
                best = (i, s);
            }
        }
        Some(best.0)
    }

    /// Every offset where `pattern` occurs exactly in `signal`, found by checking
    /// the offsets whose correlation peaks at the pattern's energy.
    pub fn find_all(&self, signal: &[f32], pattern: &[f32]) -> Vec<usize> {
        let k = pattern.len();
        let energy: f32 = pattern.iter().map(|v| v * v).sum();
        self.correlate(signal, pattern)
            .into_iter()
            .enumerate()
            .filter(|&(offset, score)| {
                // A correlation equal to the energy is necessary; confirm exactness.
                (score - energy).abs() < 1e-3 && signal[offset..offset + k] == *pattern
            })
            .map(|(offset, _)| offset)
            .collect()
    }
}
