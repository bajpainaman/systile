//! `TensorEditDistance` — Levenshtein distance as a tropical (min-plus) matmul.
//!
//! Edit distance is a shortest path. Lay the two strings out as a grid where node
//! `(i, j)` means "aligned `a[..i]` with `b[..j]`", and add edges:
//!
//! - `(i, j) → (i+1, j)` cost 1 (delete `a[i]`),
//! - `(i, j) → (i, j+1)` cost 1 (insert `b[j]`),
//! - `(i, j) → (i+1, j+1)` cost 0 if `a[i] == b[j]` else 1 (match / substitute).
//!
//! The edit distance is then the shortest path from `(0,0)` to `(m,n)`. Shortest
//! paths over the **tropical semiring** (`⊕ = min`, `⊗ = +`) are matrix products,
//! so relaxing the whole grid is a sequence of tropical vector–matrix matmuls
//! ([`crate::semiring`]) — Bellman–Ford expressed as min-plus GEMV.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;
use crate::semiring::{semiring_matmul, Tropical};

/// Levenshtein edit distance computed by tropical matrix multiply.
#[derive(Clone, Copy, Debug)]
pub struct TensorEditDistance {
    geom: Geometry,
}

impl Default for TensorEditDistance {
    fn default() -> Self {
        TensorEditDistance {
            geom: Geometry::TPU_V,
        }
    }
}

impl TensorEditDistance {
    /// Create an edit-distance engine with the default tile geometry.
    pub fn new() -> Self {
        TensorEditDistance::default()
    }

    /// The Levenshtein distance between `a` and `b`, as the shortest path through
    /// the alignment grid, found by iterating a tropical (min-plus) matmul.
    pub fn distance(&self, a: &[u8], b: &[u8]) -> usize {
        let m = a.len();
        let n = b.len();
        let rows = m + 1;
        let cols = n + 1;
        let nodes = rows * cols;
        let id = |i: usize, j: usize| i * cols + j;
        let inf = f32::INFINITY;

        // Tropical adjacency: diagonal 0 (a node relaxes itself), edges as above.
        let mut adj = vec![inf; nodes * nodes];
        for v in 0..nodes {
            adj[v * nodes + v] = 0.0;
        }
        for i in 0..rows {
            for j in 0..cols {
                let u = id(i, j);
                if i + 1 < rows {
                    adj[u * nodes + id(i + 1, j)] = 1.0; // delete
                }
                if j + 1 < cols {
                    adj[u * nodes + id(i, j + 1)] = 1.0; // insert
                }
                if i + 1 < rows && j + 1 < cols {
                    let cost = if a[i] == b[j] { 0.0 } else { 1.0 };
                    let slot = &mut adj[u * nodes + id(i + 1, j + 1)];
                    *slot = slot.min(cost);
                }
            }
        }
        let adj_lat = PaddedTileLattice::from_dense(nodes, nodes, &adj, self.geom).unwrap();

        // Single-source tropical Bellman–Ford from (0,0): dist ⊗ adj, repeated.
        let mut dist = vec![inf; nodes];
        dist[0] = 0.0;
        // The longest simple path in the DAG is m + n edges; that many relaxations
        // is enough to converge.
        for _ in 0..(m + n) {
            let d_lat = PaddedTileLattice::from_dense(1, nodes, &dist, self.geom).unwrap();
            // (1 × N) · (N × N) over min-plus: new[j] = min_k dist[k] + adj[k][j].
            let next = semiring_matmul::<Tropical>(&d_lat, &adj_lat)
                .unwrap()
                .to_dense();
            if next == dist {
                break;
            }
            dist = next;
        }

        let target = dist[id(m, n)];
        if target.is_finite() {
            target.round() as usize
        } else {
            m.max(n)
        }
    }

    /// Convenience wrapper for `&str` inputs (compares raw bytes).
    pub fn distance_str(&self, a: &str, b: &str) -> usize {
        self.distance(a.as_bytes(), b.as_bytes())
    }
}
