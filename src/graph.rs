//! `TensorGraph` — a directed weighted graph whose algorithms are matrix powers.
//!
//! The adjacency matrix is stored as a [`PaddedTileLattice`], and classic graph
//! questions are answered by raising it to a power over the right semiring
//! ([`crate::semiring`]):
//!
//! - **Reachability / transitive closure**: boolean matrix powers.
//! - **All-pairs shortest paths**: tropical (min-plus) matrix powers.
//! - **Walk counting**: ordinary matrix powers.
//!
//! Each "power" is computed by repeated squaring, so a graph of `n` nodes needs
//! only `⌈log₂ n⌉` dense matmuls — turning a traversal into a handful of GEMMs.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;
use crate::semiring::{semiring_matmul, Boolean, Counting, Semiring, Tropical};

/// A directed, weighted graph backed by a dense adjacency lattice.
#[derive(Clone)]
pub struct TensorGraph {
    n: usize,
    geom: Geometry,
    edges: Vec<(usize, usize, f32)>,
}

impl TensorGraph {
    /// Create an `n`-node graph with no edges.
    pub fn new(n: usize) -> Self {
        TensorGraph {
            n,
            geom: Geometry::TPU_V,
            edges: Vec::new(),
        }
    }

    /// Number of nodes.
    #[inline]
    pub fn nodes(&self) -> usize {
        self.n
    }

    /// Number of edges added.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Add a weighted directed edge `u → v`.
    pub fn add_edge(&mut self, u: usize, v: usize, weight: f32) {
        assert!(u < self.n && v < self.n, "endpoints must be in range");
        self.edges.push((u, v, weight));
    }

    /// Add an unweighted directed edge `u → v` (weight `1`).
    pub fn add_edge_unweighted(&mut self, u: usize, v: usize) {
        self.add_edge(u, v, 1.0);
    }

    /// Build the boolean adjacency lattice (`1` where an edge exists), with the
    /// diagonal set so the closure is reflexive.
    fn reflexive_boolean(&self) -> PaddedTileLattice<f32> {
        let mut dense = vec![0.0f32; self.n * self.n];
        for i in 0..self.n {
            dense[i * self.n + i] = 1.0;
        }
        for &(u, v, _) in &self.edges {
            dense[u * self.n + v] = 1.0;
        }
        PaddedTileLattice::from_dense(self.n, self.n, &dense, self.geom).unwrap()
    }

    /// Build the tropical adjacency lattice: edge weights off-diagonal, `0` on the
    /// diagonal, `+∞` where there is no edge.
    fn tropical_adjacency(&self) -> PaddedTileLattice<f32> {
        let mut dense = vec![f32::INFINITY; self.n * self.n];
        for i in 0..self.n {
            dense[i * self.n + i] = 0.0;
        }
        for &(u, v, w) in &self.edges {
            let slot = &mut dense[u * self.n + v];
            *slot = slot.min(w);
        }
        PaddedTileLattice::from_dense(self.n, self.n, &dense, self.geom).unwrap()
    }

    /// Build the `0/1` adjacency lattice with no self-loops, for walk counting.
    fn plain_adjacency(&self) -> PaddedTileLattice<f32> {
        let mut dense = vec![0.0f32; self.n * self.n];
        for &(u, v, _) in &self.edges {
            dense[u * self.n + v] = 1.0;
        }
        PaddedTileLattice::from_dense(self.n, self.n, &dense, self.geom).unwrap()
    }

    /// Number of squarings needed to reach any path length up to `n`.
    fn squarings(&self) -> u32 {
        // ⌈log₂(max(1, n))⌉
        let mut steps = 0;
        let mut reach = 1usize;
        while reach < self.n.max(1) {
            reach *= 2;
            steps += 1;
        }
        steps
    }

    /// Raise `m` to a closure power over semiring `S` by repeated squaring.
    fn close<S: Semiring>(&self, mut m: PaddedTileLattice<f32>) -> PaddedTileLattice<f32> {
        for _ in 0..self.squarings() {
            m = semiring_matmul::<S>(&m, &m).unwrap();
        }
        m
    }

    /// The reflexive-transitive closure: `reachable[u][v]` is `1` iff `v` is
    /// reachable from `u`. Computed as boolean matrix powers.
    pub fn reachability(&self) -> PaddedTileLattice<f32> {
        self.close::<Boolean>(self.reflexive_boolean())
    }

    /// True if `v` is reachable from `u` (including `u == v`).
    pub fn reachable(&self, u: usize, v: usize) -> bool {
        *self.reachability().get(u, v).unwrap() != 0.0
    }

    /// All-pairs shortest path distances, computed as tropical matrix powers.
    /// Unreachable pairs hold `+∞`.
    pub fn shortest_paths(&self) -> PaddedTileLattice<f32> {
        self.close::<Tropical>(self.tropical_adjacency())
    }

    /// The shortest-path distance from `u` to `v`, or `None` if unreachable.
    pub fn distance(&self, u: usize, v: usize) -> Option<f32> {
        let d = *self.shortest_paths().get(u, v).unwrap();
        if d.is_finite() {
            Some(d)
        } else {
            None
        }
    }

    /// The number of distinct walks of exactly `k` edges from `u` to `v`, as the
    /// `k`-th ordinary matrix power of the adjacency.
    pub fn walk_counts(&self, k: usize) -> PaddedTileLattice<f32> {
        let adj = self.plain_adjacency();
        if k == 0 {
            // The 0th power is the identity.
            let mut dense = vec![0.0f32; self.n * self.n];
            for i in 0..self.n {
                dense[i * self.n + i] = 1.0;
            }
            return PaddedTileLattice::from_dense(self.n, self.n, &dense, self.geom).unwrap();
        }
        let mut acc = adj.clone();
        for _ in 1..k {
            acc = semiring_matmul::<Counting>(&acc, &adj).unwrap();
        }
        acc
    }
}

impl core::fmt::Debug for TensorGraph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "TensorGraph {{ nodes: {}, edges: {} }}",
            self.n,
            self.edges.len()
        )
    }
}
