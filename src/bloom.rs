//! `TensorBloom` — a counting Bloom filter whose membership test is a matmul.
//!
//! A Bloom filter hashes each item to `k` positions in an `m`-slot array. Insertion
//! sets those slots; a query checks whether all `k` of an item's slots are set. The
//! matmul-native form: encode an item as a 0/1 **signature** vector of length `m`,
//! and the filter is the (counting) sum of every inserted signature. Testing a
//! batch of `b` items is then one matmul:
//!
//! ```text
//! counts = Sigs (b × m) · present(filter) (m × 1)
//! item i is present  ⟺  counts[i] == (number of set bits in Sigs[i])
//! ```
//!
//! Like any Bloom filter it answers "definitely not present" or "probably present"
//! — false positives, never false negatives. Keeping counts (rather than bits) also
//! lets it support removal.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;

#[inline]
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A counting Bloom filter with a matmul membership test.
#[derive(Clone)]
pub struct TensorBloom {
    m: usize,
    k: usize,
    seed: u64,
    geom: Geometry,
    counts: Vec<f32>,
    inserted: usize,
}

impl TensorBloom {
    /// Create a filter with `m` slots and `k` hash functions.
    pub fn new(m: usize, k: usize, seed: u64) -> Self {
        assert!(m > 0 && k > 0, "m and k must be positive");
        TensorBloom {
            m,
            k,
            seed,
            geom: Geometry::TPU_V,
            counts: vec![0.0; m],
            inserted: 0,
        }
    }

    /// Number of slots.
    #[inline]
    pub fn slots(&self) -> usize {
        self.m
    }

    /// Number of hash functions.
    #[inline]
    pub fn hashes(&self) -> usize {
        self.k
    }

    /// Number of insertions performed.
    #[inline]
    pub fn inserted(&self) -> usize {
        self.inserted
    }

    /// The `k` (possibly colliding) slot positions for an item, by double hashing.
    fn positions(&self, item: u64) -> Vec<usize> {
        let h1 = splitmix64(item ^ self.seed);
        let h2 = splitmix64(item ^ self.seed.rotate_left(32) ^ 0xDEAD_BEEF);
        let mut out = Vec::with_capacity(self.k);
        for i in 0..self.k as u64 {
            out.push((h1.wrapping_add(i.wrapping_mul(h2)) % self.m as u64) as usize);
        }
        out
    }

    /// The distinct slot positions for an item (collisions removed).
    fn distinct_positions(&self, item: u64) -> Vec<usize> {
        let mut p = self.positions(item);
        p.sort_unstable();
        p.dedup();
        p
    }

    /// Insert an item.
    pub fn insert(&mut self, item: u64) {
        for pos in self.distinct_positions(item) {
            self.counts[pos] += 1.0;
        }
        self.inserted += 1;
    }

    /// Remove an item (counting filters support deletion).
    pub fn remove(&mut self, item: u64) {
        for pos in self.distinct_positions(item) {
            if self.counts[pos] > 0.0 {
                self.counts[pos] -= 1.0;
            }
        }
        self.inserted = self.inserted.saturating_sub(1);
    }

    /// Scalar membership test: probably-present (`true`) or definitely-absent.
    pub fn contains(&self, item: u64) -> bool {
        self.distinct_positions(item)
            .into_iter()
            .all(|pos| self.counts[pos] > 0.0)
    }

    /// The boolean presence vector (`1` where a slot is non-empty) as an `m × 1`
    /// lattice — the right operand of the membership matmul.
    fn present_vector(&self) -> PaddedTileLattice<f32> {
        let present: Vec<f32> = self
            .counts
            .iter()
            .map(|&c| if c > 0.0 { 1.0 } else { 0.0 })
            .collect();
        PaddedTileLattice::from_dense(self.m, 1, &present, self.geom).unwrap()
    }

    /// Test a batch of items with a single matmul of their signatures against the
    /// filter's presence vector.
    pub fn batch_contains(&self, items: &[u64]) -> Vec<bool> {
        let b = items.len();
        if b == 0 {
            return Vec::new();
        }
        let mut sigs = vec![0.0f32; b * self.m];
        let mut needed = Vec::with_capacity(b);
        for (i, &item) in items.iter().enumerate() {
            let positions = self.distinct_positions(item);
            needed.push(positions.len() as f32);
            for pos in positions {
                sigs[i * self.m + pos] = 1.0;
            }
        }
        let sig_lat = PaddedTileLattice::from_dense(b, self.m, &sigs, self.geom).unwrap();
        // (b × m) · (m × 1) → (b × 1): how many of each item's slots are set.
        let counts = sig_lat.matmul(&self.present_vector()).unwrap().to_dense();
        counts
            .iter()
            .zip(&needed)
            .map(|(&got, &need)| got >= need)
            .collect()
    }

    /// The textbook false-positive probability `(1 − e^{−kn/m})^k` for the current
    /// load.
    pub fn estimated_fpr(&self) -> f64 {
        let exponent = -(self.k as f64) * self.inserted as f64 / self.m as f64;
        (1.0 - exponent.exp()).powi(self.k as i32)
    }
}

impl core::fmt::Debug for TensorBloom {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "TensorBloom {{ m: {}, k: {}, inserted: {}, fpr~{:.4} }}",
            self.m,
            self.k,
            self.inserted,
            self.estimated_fpr()
        )
    }
}
