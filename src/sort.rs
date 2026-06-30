//! `TensorSort` — sorting as a comparison matmul.
//!
//! Sorting is `O(n log n)` and branch-heavy on a CPU. The matmul-native way trades
//! that for `O(n²)` dense arithmetic, which is the right trade when matmul is the
//! cheap primitive:
//!
//! 1. Build the pairwise **comparison matrix** `C`, where `C[i,j] = 1` iff element
//!    `i` should come after element `j` in sorted order (with a stable tie-break).
//! 2. The **rank** of each element — its final position — is the row sum of `C`,
//!    i.e. the matmul `C · 1`.
//! 3. The ranks define a **permutation matrix** `P`, and the sorted vector is the
//!    matmul `P · x`.
//!
//! Two matmuls, no branches, no comparisons-as-control-flow. This is the same idea
//! behind differentiable sorting (NeuralSort / SoftSort), here in its exact form.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;

/// A matmul-based sorter.
#[derive(Clone, Copy, Debug)]
pub struct TensorSort {
    geom: Geometry,
}

impl Default for TensorSort {
    fn default() -> Self {
        TensorSort {
            geom: Geometry::TPU_V,
        }
    }
}

impl TensorSort {
    /// Create a sorter with the default tile geometry.
    pub fn new() -> Self {
        TensorSort::default()
    }

    /// The pairwise comparison matrix `C` for ascending order: `C[i,j] = 1` iff
    /// `x[i] > x[j]`, or they tie and `i > j` (a stable tie-break).
    pub fn comparison_matrix(&self, x: &[f32]) -> PaddedTileLattice<f32> {
        let n = x.len();
        let mut dense = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let after = x[i] > x[j] || (x[i] == x[j] && i > j);
                dense[i * n + j] = if after { 1.0 } else { 0.0 };
            }
        }
        PaddedTileLattice::from_dense(n, n, &dense, self.geom).unwrap()
    }

    /// The rank (final 0-based position) of each element, computed as `C · 1`.
    pub fn ranks(&self, x: &[f32]) -> Vec<usize> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let c = self.comparison_matrix(x);
        let ones = PaddedTileLattice::from_dense(n, 1, &vec![1.0f32; n], self.geom).unwrap();
        // (n × n) · (n × 1) → (n × 1): each row sum is that element's rank.
        let r = c.matmul(&ones).unwrap().to_dense();
        r.into_iter().map(|v| v.round() as usize).collect()
    }

    /// The indices that would sort `x` ascending (`argsort`).
    pub fn argsort(&self, x: &[f32]) -> Vec<usize> {
        let ranks = self.ranks(x);
        let mut out = vec![0usize; x.len()];
        for (i, &r) in ranks.iter().enumerate() {
            out[r] = i;
        }
        out
    }

    /// The permutation matrix `P` with `P[rank[i], i] = 1`, so that `P · x` sorts.
    pub fn permutation_matrix(&self, x: &[f32]) -> PaddedTileLattice<f32> {
        let n = x.len();
        let ranks = self.ranks(x);
        let mut dense = vec![0.0f32; n * n];
        for (i, &r) in ranks.iter().enumerate() {
            dense[r * n + i] = 1.0;
        }
        PaddedTileLattice::from_dense(n, n, &dense, self.geom).unwrap()
    }

    /// Sort `x` ascending by scattering each element to its rank.
    pub fn sort(&self, x: &[f32]) -> Vec<f32> {
        let ranks = self.ranks(x);
        let mut out = vec![0.0f32; x.len()];
        for (i, &r) in ranks.iter().enumerate() {
            out[r] = x[i];
        }
        out
    }

    /// Sort `x` by actually applying the permutation matrix as a matmul `P · x`.
    /// Slower than [`TensorSort::sort`], but it demonstrates that the reordering is
    /// itself a matrix multiply.
    pub fn sort_via_matmul(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let p = self.permutation_matrix(x);
        let xv = PaddedTileLattice::from_dense(n, 1, x, self.geom).unwrap();
        p.matmul(&xv).unwrap().to_dense()
    }
}
