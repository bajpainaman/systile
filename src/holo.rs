//! `HoloMemory` — a key→value store that lives in superposition inside a single
//! hypervector, and whose batch lookup is one matrix multiply.
//!
//! # The idea
//!
//! A conventional map stores `n` entries in `n` slots and finds one by hashing or
//! comparing. A holographic map stores *all* `n` entries **summed on top of each
//! other** in one fixed-width vector, and finds one by algebra:
//!
//! ```text
//! insert(k, v):   memory += atom(k) ⊛ atom(v)          // bind then bundle
//! get(k):         noisy   = memory ⊛ atom(k)            // unbind
//!                 value   = argmax_j  noisy · codebook[:, j]   // cleanup = matmul
//! ```
//!
//! Unbinding cancels the key back out (`atom(k) ⊛ atom(k) = 1`), leaving the bound
//! value plus a cloud of near-orthogonal noise from the *other* entries. The
//! cleanup step projects that noisy vector onto the value codebook and takes the
//! best match — and for a whole batch of keys that projection is a single
//! `(b × dim) · (dim × m)` matmul on the systolic engine ([`Codebook::cleanup_batch`]).
//!
//! # Why this is a TPU structure
//!
//! On a CPU this is a terrible map: a hash table is `O(1)` and exact, while this is
//! a giant dense matmul that only *probably* returns the right value. The trade
//! only pays off where dense matmul is the cheap primitive and branch-y pointer
//! chasing is expensive — i.e. a TPU — and where you want to resolve thousands of
//! lookups at once, tolerate noise, or keep the whole structure differentiable.
//! It is not that this is *impossible* elsewhere; it is that nowhere else is it the
//! *right* shape.

use crate::codebook::Codebook;
use crate::hyper::Hyper;

/// A holographic key→value associative store.
///
/// Keys and values are symbol ids in `0..n_keys` and `0..n_values`. The entire map
/// is the single `memory` hypervector; nothing else grows as entries are inserted.
#[derive(Clone)]
pub struct HoloMemory {
    dim: usize,
    key_seed: u64,
    values: Codebook,
    memory: Hyper,
    len: usize,
}

impl HoloMemory {
    /// Create an empty map over a `dim`-dimensional space, with `n_values`
    /// distinct value symbols. `seed` makes the whole structure reproducible.
    pub fn new(dim: usize, n_values: usize, seed: u64) -> Self {
        // Namespace the key and value codebooks so their atoms are independent.
        let key_seed = seed ^ 0x4B45_5953_4545_4421;
        let value_seed = seed ^ 0x5641_4C55_4553_2121;
        HoloMemory {
            dim,
            key_seed,
            values: Codebook::new(dim, n_values, value_seed),
            memory: Hyper::zeros(dim),
            len: 0,
        }
    }

    /// Hypervector dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of entries bundled in so far (not a slot count — storage is fixed).
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if nothing has been inserted.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the underlying superposition vector — the entire stored map.
    #[inline]
    pub fn memory(&self) -> &Hyper {
        &self.memory
    }

    /// The value codebook used for cleanup.
    #[inline]
    pub fn values(&self) -> &Codebook {
        &self.values
    }

    /// Regenerate the atomic hypervector for key id `key`.
    pub fn key_atom(&self, key: usize) -> Hyper {
        Hyper::atom(self.dim, self.key_seed, key as u64)
    }

    /// Insert (or add weight to) the binding `key → value`.
    pub fn insert(&mut self, key: usize, value: usize) {
        let bound = self.key_atom(key).bind(&self.values.atom(value));
        self.memory.bundle_into(&bound);
        self.len += 1;
    }

    /// Remove one unit of the binding `key → value`. Removing a binding that was
    /// never inserted simply pushes the memory in the opposite direction.
    pub fn remove(&mut self, key: usize, value: usize) {
        let bound = self.key_atom(key).bind(&self.values.atom(value));
        self.memory.unbundle(&bound);
        self.len = self.len.saturating_sub(1);
    }

    /// Unbind `key` from the memory, yielding the noisy value hypervector that
    /// cleanup will resolve.
    pub fn probe(&self, key: usize) -> Hyper {
        self.memory.bind(&self.key_atom(key))
    }

    /// Look up the value bound to `key`, returning `(value_id, score)`. The score
    /// is the raw dot product against the winning value atom; a higher score means
    /// a more confident recall.
    pub fn get(&self, key: usize) -> (usize, f32) {
        self.values.cleanup(&self.probe(key))
    }

    /// Look up many keys at once. All the cleanups happen in a single systolic
    /// matmul, which is the operation this whole structure is built around.
    pub fn batch_get(&self, keys: &[usize]) -> Vec<(usize, f32)> {
        let probes: Vec<Hyper> = keys.iter().map(|&k| self.probe(k)).collect();
        self.values.cleanup_batch(&probes)
    }

    /// A rough estimate of how many bindings this dimension can hold before recall
    /// degrades, using the standard `dim / (2 ln m)` capacity rule of thumb.
    pub fn estimated_capacity(&self) -> f64 {
        let m = self.values.len().max(2) as f64;
        self.dim as f64 / (2.0 * m.ln())
    }

    /// The current load relative to [`HoloMemory::estimated_capacity`]; values
    /// above `1.0` mean recall is expected to start failing.
    pub fn load_factor(&self) -> f64 {
        let cap = self.estimated_capacity();
        if cap == 0.0 {
            f64::INFINITY
        } else {
            self.len as f64 / cap
        }
    }
}

impl core::fmt::Debug for HoloMemory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "HoloMemory {{ dim: {}, entries: {}, values: {}, load: {:.2} }}",
            self.dim,
            self.len,
            self.values.len(),
            self.load_factor()
        )
    }
}
