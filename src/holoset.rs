//! `HoloSet` — a set held in superposition, with membership testing as a matmul.
//!
//! Where [`crate::holo::HoloMemory`] bundles *bound pairs*, a `HoloSet` bundles
//! bare symbols: `memory = Σ atom(id)`. Membership is then a dot product —
//! `memory · atom(q)` is large when `q` is in the set and near zero otherwise — so
//! testing a batch of candidates against the set is one matmul against the
//! codebook. Union is just bundling two sets together, and the set's size can be
//! read straight off the squared norm.

use crate::codebook::Codebook;
use crate::hyper::Hyper;

/// A set of symbol ids stored in one superposition vector.
#[derive(Clone)]
pub struct HoloSet {
    codebook: Codebook,
    memory: Hyper,
    len: usize,
}

impl HoloSet {
    /// Create an empty set over a `dim`-dimensional space with `universe` possible
    /// member symbols.
    pub fn new(dim: usize, universe: usize, seed: u64) -> Self {
        let codebook = Codebook::new(dim, universe, seed);
        let memory = Hyper::zeros(dim);
        HoloSet {
            codebook,
            memory,
            len: 0,
        }
    }

    /// Hypervector dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.codebook.dim()
    }

    /// Number of `insert` calls (not deduplicated).
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if nothing has been inserted.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The underlying superposition vector — the whole set in one tensor.
    #[inline]
    pub fn memory(&self) -> &Hyper {
        &self.memory
    }

    /// Add a symbol to the set.
    pub fn insert(&mut self, id: usize) {
        self.memory.bundle_into(&self.codebook.atom(id));
        self.len += 1;
    }

    /// Normalised membership score for `id`, near `1.0` for members and near `0.0`
    /// for non-members.
    pub fn similarity(&self, id: usize) -> f32 {
        self.memory.dot(&self.codebook.atom(id)) / self.dim() as f32
    }

    /// Test membership against the standard `0.5` decision threshold.
    pub fn contains(&self, id: usize) -> bool {
        self.similarity(id) > 0.5
    }

    /// Test a batch of candidate ids in one matmul, returning a membership flag for
    /// each. The set vector is scored against the whole codebook at once.
    pub fn batch_contains(&self, ids: &[usize]) -> Vec<bool> {
        let scores = self
            .codebook
            .scores_batch(core::slice::from_ref(&self.memory));
        let row = &scores[0];
        let threshold = 0.5 * self.dim() as f32;
        ids.iter().map(|&id| row[id] > threshold).collect()
    }

    /// Estimate the number of distinct members from the squared norm: a sum of `k`
    /// near-orthogonal bipolar vectors has squared norm `≈ k · dim`.
    pub fn estimated_cardinality(&self) -> f64 {
        let n = self.memory.dot(&self.memory) as f64;
        n / self.dim() as f64
    }

    /// The union of two sets that share the same codebook, by bundling their
    /// memories. Panics if the codebooks differ in size or seed.
    pub fn union(&self, other: &HoloSet) -> HoloSet {
        assert_eq!(self.dim(), other.dim(), "sets must share a dimension");
        assert_eq!(
            self.codebook.seed(),
            other.codebook.seed(),
            "sets must share a codebook"
        );
        let mut memory = self.memory.clone();
        memory.bundle_into(&other.memory);
        HoloSet {
            codebook: self.codebook.clone(),
            memory,
            len: self.len + other.len,
        }
    }
}

impl core::fmt::Debug for HoloSet {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "HoloSet {{ dim: {}, inserts: {}, ~card: {:.1} }}",
            self.dim(),
            self.len,
            self.estimated_cardinality()
        )
    }
}
