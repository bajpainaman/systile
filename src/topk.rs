//! `TensorTopK` — selecting the `k` largest as a comparison-count matmul.
//!
//! To find the top `k` elements without sorting everything, count for each element
//! how many others beat it: that count *is* its descending rank, and the top `k`
//! are exactly the elements whose count is below `k`. The count is the matmul
//! `C · 1`, where `C[i,j] = 1` iff element `j` outranks element `i` (with a stable
//! tie-break). One matmul ranks the whole vector; selection is then a threshold.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;

/// A matmul-based top-k selector.
#[derive(Clone, Copy, Debug)]
pub struct TensorTopK {
    geom: Geometry,
}

impl Default for TensorTopK {
    fn default() -> Self {
        TensorTopK {
            geom: Geometry::TPU_V,
        }
    }
}

impl TensorTopK {
    /// Create a selector with the default tile geometry.
    pub fn new() -> Self {
        TensorTopK::default()
    }

    /// The descending rank of every element: `rank[i]` = number of elements that
    /// outrank `i`, computed as `C · 1`. The largest element has rank `0`.
    pub fn ranks(&self, scores: &[f32]) -> Vec<usize> {
        let n = scores.len();
        if n == 0 {
            return Vec::new();
        }
        let mut dense = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                // j outranks i if it is larger, or equal with a smaller index.
                let above = scores[j] > scores[i] || (scores[j] == scores[i] && j < i);
                if above {
                    dense[i * n + j] = 1.0;
                }
            }
        }
        let c = PaddedTileLattice::from_dense(n, n, &dense, self.geom).unwrap();
        let ones = PaddedTileLattice::from_dense(n, 1, &vec![1.0f32; n], self.geom).unwrap();
        c.matmul(&ones)
            .unwrap()
            .to_dense()
            .into_iter()
            .map(|v| v.round() as usize)
            .collect()
    }

    /// The top `k` elements as `(index, score)` pairs, ordered best first.
    pub fn select(&self, scores: &[f32], k: usize) -> Vec<(usize, f32)> {
        let ranks = self.ranks(scores);
        let mut chosen: Vec<(usize, f32)> = ranks
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r < k)
            .map(|(i, _)| (i, scores[i]))
            .collect();
        // Order by rank (descending score) using the computed ranks.
        chosen.sort_by_key(|&(i, _)| ranks[i]);
        chosen
    }

    /// The indices of the top `k` elements, best first.
    pub fn select_indices(&self, scores: &[f32], k: usize) -> Vec<usize> {
        self.select(scores, k).into_iter().map(|(i, _)| i).collect()
    }

    /// Select the top `k` from each row of a batch of score vectors.
    pub fn select_batch(&self, rows: &[&[f32]], k: usize) -> Vec<Vec<(usize, f32)>> {
        rows.iter().map(|row| self.select(row, k)).collect()
    }

    /// The `k`-th largest value (the selection threshold), or `None` if `k` is `0`
    /// or out of range.
    pub fn kth_largest(&self, scores: &[f32], k: usize) -> Option<f32> {
        if k == 0 || k > scores.len() {
            return None;
        }
        let ranks = self.ranks(scores);
        ranks.iter().position(|&r| r == k - 1).map(|i| scores[i])
    }
}
