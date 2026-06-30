//! `TensorIndex` — exact nearest-neighbour search, where the search is a matmul.
//!
//! This is the vector-database workload in its most honest form. Store `n` item
//! vectors as the rows of a matrix; to find the items most similar to a batch of
//! `b` query vectors, multiply `Q (b × dim) · Cᵀ (dim × n)` and read the top
//! scores off each row. No trees, no quantisation, no approximation — one dense
//! matmul scores every query against every item, which is exactly the regime a
//! systolic matrix unit is built for.
//!
//! Unlike [`crate::holoset::HoloSet`], which stores items in superposition and
//! answers *approximately*, a `TensorIndex` keeps each vector intact and returns
//! the *exact* nearest neighbours.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;

/// A scored search hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// The id of the matched item (its insertion order).
    pub id: usize,
    /// The similarity score (dot product with the query).
    pub score: f32,
}

/// An exact nearest-neighbour index over fixed-dimension vectors.
#[derive(Clone)]
pub struct TensorIndex {
    dim: usize,
    geom: Geometry,
    items: Vec<Vec<f32>>,
}

impl TensorIndex {
    /// Create an empty index over `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        TensorIndex {
            dim,
            geom: Geometry::TPU_V,
            items: Vec::new(),
        }
    }

    /// Vector dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of indexed items.
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True if the index holds no items.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Add a vector to the index, returning its assigned id.
    pub fn add(&mut self, vector: Vec<f32>) -> usize {
        assert_eq!(vector.len(), self.dim, "vector dimension must match index");
        let id = self.items.len();
        self.items.push(vector);
        id
    }

    /// Borrow the stored vector for `id`.
    pub fn vector(&self, id: usize) -> &[f32] {
        &self.items[id]
    }

    /// Build the `dim × n` corpus matrix (column `j` is item `j`).
    fn corpus_matrix(&self) -> PaddedTileLattice<f32> {
        let n = self.items.len();
        let mut dense = vec![0.0f32; self.dim * n];
        for (j, item) in self.items.iter().enumerate() {
            for (p, &value) in item.iter().enumerate() {
                dense[p * n + j] = value;
            }
        }
        PaddedTileLattice::from_dense(self.dim, n, &dense, self.geom).unwrap()
    }

    /// Score every query against every item in one matmul, returning a row of `n`
    /// similarity scores per query.
    pub fn score_batch(&self, queries: &[&[f32]]) -> Vec<Vec<f32>> {
        let b = queries.len();
        if b == 0 || self.items.is_empty() {
            return vec![Vec::new(); b];
        }
        let n = self.items.len();
        let mut qdense = vec![0.0f32; b * self.dim];
        for (i, q) in queries.iter().enumerate() {
            assert_eq!(q.len(), self.dim, "query dimension must match index");
            qdense[i * self.dim..(i + 1) * self.dim].copy_from_slice(q);
        }
        let qlat = PaddedTileLattice::from_dense(b, self.dim, &qdense, self.geom).unwrap();
        let scores = qlat.matmul(&self.corpus_matrix()).unwrap().to_dense();
        scores.chunks_exact(n).map(|row| row.to_vec()).collect()
    }

    /// The top-`k` nearest items to a batch of queries, by descending score.
    pub fn search_batch(&self, queries: &[&[f32]], k: usize) -> Vec<Vec<Hit>> {
        self.score_batch(queries)
            .into_iter()
            .map(|row| {
                let mut hits: Vec<Hit> = row
                    .into_iter()
                    .enumerate()
                    .map(|(id, score)| Hit { id, score })
                    .collect();
                hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                hits.truncate(k);
                hits
            })
            .collect()
    }

    /// The top-`k` nearest items to a single query.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        self.search_batch(&[query], k).pop().unwrap_or_default()
    }

    /// The single nearest item to a query, or `None` if the index is empty.
    pub fn nearest(&self, query: &[f32]) -> Option<Hit> {
        self.search(query, 1).into_iter().next()
    }
}

impl core::fmt::Debug for TensorIndex {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "TensorIndex {{ dim: {}, items: {} }}",
            self.dim,
            self.items.len()
        )
    }
}
