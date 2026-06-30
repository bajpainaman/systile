//! A codebook: a vocabulary of atomic symbols stored as a tile-aligned matrix so
//! that *cleanup* — finding the symbol most similar to a noisy hypervector — is a
//! single matrix multiply on the systolic engine.
//!
//! The codebook holds its `count` symbols as the columns of a `dim × count`
//! [`PaddedTileLattice`]. Resolving a batch of `b` query hypervectors is then the
//! product `Q (b × dim) · C (dim × count) → S (b × count)`, after which each row's
//! `argmax` names the recovered symbol. That product is exactly what a TPU's
//! matrix unit exists to do, and it resolves the *entire* batch against the
//! *entire* vocabulary in one shot.

use crate::geometry::Geometry;
use crate::hyper::Hyper;
use crate::lattice::PaddedTileLattice;

/// A vocabulary of `count` deterministic bipolar hypervectors of dimension `dim`.
#[derive(Clone)]
pub struct Codebook {
    dim: usize,
    count: usize,
    seed: u64,
    geom: Geometry,
    /// Shape `dim × count`: column `j` is the atom for symbol `j`.
    matrix: PaddedTileLattice<f32>,
}

impl Codebook {
    /// Build a codebook of `count` symbols, each a `dim`-dimensional bipolar atom
    /// drawn deterministically from `seed`.
    pub fn new(dim: usize, count: usize, seed: u64) -> Self {
        Codebook::with_geometry(dim, count, seed, Geometry::TPU_V)
    }

    /// Build a codebook with an explicit tile geometry.
    pub fn with_geometry(dim: usize, count: usize, seed: u64, geom: Geometry) -> Self {
        let mut dense = vec![0.0f32; dim * count];
        for j in 0..count {
            let atom = Hyper::atom(dim, seed, j as u64);
            let coords = atom.as_slice();
            for (p, &value) in coords.iter().enumerate() {
                dense[p * count + j] = value;
            }
        }
        let matrix = PaddedTileLattice::from_dense(dim, count, &dense, geom)
            .expect("dense buffer is exactly dim*count");
        Codebook {
            dim,
            count,
            seed,
            geom,
            matrix,
        }
    }

    /// Hypervector dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of symbols in the vocabulary.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// True if the codebook has no symbols.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The seed this codebook was generated from.
    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Borrow the `dim × count` symbol matrix (column `j` is symbol `j`).
    #[inline]
    pub fn matrix(&self) -> &PaddedTileLattice<f32> {
        &self.matrix
    }

    /// Regenerate symbol `id` as a hypervector.
    pub fn atom(&self, id: usize) -> Hyper {
        Hyper::atom(self.dim, self.seed, id as u64)
    }

    /// Resolve one noisy hypervector to its nearest symbol, returning
    /// `(symbol_id, score)`. This is the matrix-vector form of cleanup.
    pub fn cleanup(&self, query: &Hyper) -> (usize, f32) {
        let mut best = (0usize, f32::NEG_INFINITY);
        let scores = self.scores_batch(core::slice::from_ref(query));
        for (j, &s) in scores[0].iter().enumerate() {
            if s > best.1 {
                best = (j, s);
            }
        }
        best
    }

    /// Resolve a batch of noisy hypervectors against the whole vocabulary with a
    /// single systolic matmul, returning the winning `(symbol_id, score)` per query.
    pub fn cleanup_batch(&self, queries: &[Hyper]) -> Vec<(usize, f32)> {
        let scores = self.scores_batch(queries);
        scores
            .into_iter()
            .map(|row| {
                let mut best = (0usize, f32::NEG_INFINITY);
                for (j, s) in row.into_iter().enumerate() {
                    if s > best.1 {
                        best = (j, s);
                    }
                }
                best
            })
            .collect()
    }

    /// The raw `b × count` similarity scores for a batch of queries, computed as
    /// the matmul `Q · C` on the systolic engine.
    pub fn scores_batch(&self, queries: &[Hyper]) -> Vec<Vec<f32>> {
        let b = queries.len();
        if b == 0 {
            return Vec::new();
        }
        let mut qdense = vec![0.0f32; b * self.dim];
        for (i, q) in queries.iter().enumerate() {
            debug_assert_eq!(q.dim(), self.dim, "query dim must match codebook dim");
            qdense[i * self.dim..(i + 1) * self.dim].copy_from_slice(q.as_slice());
        }
        let qlat = PaddedTileLattice::from_dense(b, self.dim, &qdense, self.geom)
            .expect("query buffer is exactly b*dim");
        // The one matmul: (b × dim) · (dim × count) → (b × count).
        let product = qlat
            .matmul(&self.matrix)
            .expect("contraction dim and geometry match by construction");
        let dense = product.to_dense();
        dense
            .chunks_exact(self.count)
            .map(|row| row.to_vec())
            .collect()
    }

    /// Re-superpose the codebook by a weight per symbol: `X · w`, a `dim`-vector.
    ///
    /// This is the second matmul of a cleanup projection: given similarity weights
    /// (e.g. from [`Codebook::scores_batch`]), it rebuilds the weighted sum of
    /// atoms. Together, `superpose(scores_batch(v))` is the projection `X Xᵀ v`
    /// that a resonator network iterates.
    pub fn superpose(&self, weights: &[f32]) -> Hyper {
        assert_eq!(weights.len(), self.count, "one weight per symbol");
        let wlat = PaddedTileLattice::from_dense(self.count, 1, weights, self.geom)
            .expect("weight buffer is exactly count*1");
        // (dim × count) · (count × 1) → (dim × 1).
        let product = self
            .matrix
            .matmul(&wlat)
            .expect("contraction dim and geometry match by construction");
        Hyper::from_vec(product.to_dense())
    }
}

impl core::fmt::Debug for Codebook {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Codebook {{ dim: {}, symbols: {} }}",
            self.dim, self.count
        )
    }
}
