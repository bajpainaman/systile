//! `CountMinSketch` — frequency estimation as a matmul per hash row.
//!
//! A Count-Min sketch keeps a `d × w` table of counters. Each item hashes to one
//! column per row; inserting bumps those `d` counters, and the frequency estimate
//! is the minimum of them (the row least corrupted by collisions). The
//! matmul-native batch query: for each row, encode the batch's chosen columns as a
//! one-hot selection matrix and multiply by that row's counters —
//!
//! ```text
//! row_counts_r = Sel_r (b × w) · counters_r (w × 1)
//! estimate[i]  = min over rows of row_counts_r[i]
//! ```
//!
//! so `d` small matmuls estimate the frequencies of a whole batch. Count-Min never
//! underestimates.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;

#[inline]
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A Count-Min sketch with a matmul batch-estimate.
#[derive(Clone)]
pub struct CountMinSketch {
    d: usize,
    w: usize,
    seed: u64,
    geom: Geometry,
    table: Vec<f32>,
    total: u64,
}

impl CountMinSketch {
    /// Create a sketch with `d` rows (hash functions) and `w` columns per row.
    pub fn new(d: usize, w: usize, seed: u64) -> Self {
        assert!(d > 0 && w > 0, "d and w must be positive");
        CountMinSketch {
            d,
            w,
            seed,
            geom: Geometry::TPU_V,
            table: vec![0.0; d * w],
            total: 0,
        }
    }

    /// Number of hash rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.d
    }

    /// Number of columns per row.
    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }

    /// Total count inserted across all items.
    #[inline]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// The column item `item` maps to in row `r`.
    fn column(&self, r: usize, item: u64) -> usize {
        (splitmix64(item ^ self.seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            % self.w as u64) as usize
    }

    /// Add `n` to the count of `item`.
    pub fn add(&mut self, item: u64, n: u64) {
        for r in 0..self.d {
            let c = self.column(r, item);
            self.table[r * self.w + c] += n as f32;
        }
        self.total += n;
    }

    /// Insert one occurrence of `item`.
    pub fn insert(&mut self, item: u64) {
        self.add(item, 1);
    }

    /// Estimate the count of `item` (never an underestimate).
    pub fn estimate(&self, item: u64) -> f32 {
        (0..self.d)
            .map(|r| self.table[r * self.w + self.column(r, item)])
            .fold(f32::INFINITY, f32::min)
    }

    /// Estimate a batch of items, taking the min over `d` matmuls — one per row.
    pub fn batch_estimate(&self, items: &[u64]) -> Vec<f32> {
        let b = items.len();
        if b == 0 {
            return Vec::new();
        }
        let mut est = vec![f32::INFINITY; b];
        for r in 0..self.d {
            // One-hot selection of each item's column in this row.
            let mut sel = vec![0.0f32; b * self.w];
            for (i, &item) in items.iter().enumerate() {
                sel[i * self.w + self.column(r, item)] = 1.0;
            }
            let sel_lat = PaddedTileLattice::from_dense(b, self.w, &sel, self.geom).unwrap();
            let row = &self.table[r * self.w..(r + 1) * self.w];
            let row_lat = PaddedTileLattice::from_dense(self.w, 1, row, self.geom).unwrap();
            // (b × w) · (w × 1) → (b × 1): this row's counter for each item.
            let counts = sel_lat.matmul(&row_lat).unwrap().to_dense();
            for (e, &c) in est.iter_mut().zip(&counts) {
                *e = e.min(c);
            }
        }
        est
    }
}

impl core::fmt::Debug for CountMinSketch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "CountMinSketch {{ d: {}, w: {}, total: {} }}",
            self.d, self.w, self.total
        )
    }
}
