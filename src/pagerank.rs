//! `TensorPageRank` — PageRank as power iteration.
//!
//! PageRank is the stationary distribution of the "random surfer" Markov chain on
//! a graph: with probability `α` follow a random out-link, otherwise teleport to a
//! uniformly random node. That distribution is the dominant eigenvector of the
//! Google matrix `M = αP + (1−α)·(1/n)·11ᵀ`, and the textbook way to find it is
//! **power iteration**: start from a uniform vector and repeatedly apply `M`, i.e.
//! repeated matrix–vector multiplies until the ranks stop moving.
//!
//! Each iteration is one matmul on the systolic engine, so the whole algorithm is a
//! short chain of GEMVs.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;

/// PageRank over a directed graph, by power iteration.
#[derive(Clone)]
pub struct TensorPageRank {
    n: usize,
    geom: Geometry,
    out: Vec<Vec<usize>>,
}

impl TensorPageRank {
    /// Create an `n`-node graph with no links.
    pub fn new(n: usize) -> Self {
        TensorPageRank {
            n,
            geom: Geometry::TPU_V,
            out: vec![Vec::new(); n],
        }
    }

    /// Number of nodes.
    #[inline]
    pub fn nodes(&self) -> usize {
        self.n
    }

    /// Add a directed link `from → to`.
    pub fn add_link(&mut self, from: usize, to: usize) {
        assert!(from < self.n && to < self.n, "endpoints must be in range");
        self.out[from].push(to);
    }

    /// Build the column-stochastic transition matrix `P` as a `dim × dim` lattice.
    /// A dangling node (no out-links) spreads its mass uniformly so columns sum to 1.
    fn transition_matrix(&self) -> PaddedTileLattice<f32> {
        let n = self.n;
        let mut dense = vec![0.0f32; n * n];
        for (from, links) in self.out.iter().enumerate() {
            if links.is_empty() {
                // Dangling: uniform column so rank mass is conserved.
                for to in 0..n {
                    dense[to * n + from] += 1.0 / n as f32;
                }
            } else {
                let share = 1.0 / links.len() as f32;
                for &to in links {
                    dense[to * n + from] += share;
                }
            }
        }
        PaddedTileLattice::from_dense(n, n, &dense, self.geom).unwrap()
    }

    /// Run power iteration and return the PageRank vector (sums to 1). `alpha` is
    /// the damping factor (typically `0.85`); iteration stops at `max_iters` or when
    /// the L1 change falls below `tol`.
    pub fn rank(&self, alpha: f32, max_iters: usize, tol: f32) -> Vec<f32> {
        let n = self.n;
        if n == 0 {
            return Vec::new();
        }
        let p = self.transition_matrix();
        let teleport = (1.0 - alpha) / n as f32;
        let mut r = vec![1.0f32 / n as f32; n];

        for _ in 0..max_iters {
            // One matmul: follow links. r_next = alpha * P r + teleport.
            let rv = PaddedTileLattice::from_dense(n, 1, &r, self.geom).unwrap();
            let pr = p.matmul(&rv).unwrap().to_dense();
            let mut next: Vec<f32> = pr.iter().map(|&v| alpha * v + teleport).collect();
            // Renormalise to guard against floating-point drift.
            let sum: f32 = next.iter().sum();
            for v in next.iter_mut() {
                *v /= sum;
            }
            let delta: f32 = next.iter().zip(&r).map(|(a, b)| (a - b).abs()).sum();
            r = next;
            if delta < tol {
                break;
            }
        }
        r
    }

    /// Run PageRank with conventional defaults (`alpha = 0.85`, 100 iterations).
    pub fn rank_default(&self) -> Vec<f32> {
        self.rank(0.85, 100, 1e-6)
    }
}

impl core::fmt::Debug for TensorPageRank {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let edges: usize = self.out.iter().map(|o| o.len()).sum();
        write!(
            f,
            "TensorPageRank {{ nodes: {}, links: {} }}",
            self.n, edges
        )
    }
}
