//! `TensorAttention` — scaled dot-product attention as a soft retrieval memory.
//!
//! Attention is the operation modern accelerators exist to run, and it is matmuls
//! all the way down:
//!
//! ```text
//! scores  = Q · Kᵀ / √d         (Lq × Lk)
//! weights = softmax(scores)      (row-wise)
//! out     = weights · V          (Lq × dv)
//! ```
//!
//! Read as a data structure, it is a **soft associative memory**: each query
//! retrieves a convex blend of the value rows, weighted by how well it matches the
//! corresponding keys. A sharp match returns essentially one value (like
//! [`crate::index::TensorIndex`]); a diffuse one returns an average.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;

/// Scaled dot-product attention over `d`-dimensional keys and queries.
#[derive(Clone, Copy, Debug)]
pub struct TensorAttention {
    d: usize,
    geom: Geometry,
}

impl TensorAttention {
    /// Create an attention block for query/key dimension `d`.
    pub fn new(d: usize) -> Self {
        assert!(d > 0, "key dimension must be positive");
        TensorAttention {
            d,
            geom: Geometry::TPU_V,
        }
    }

    /// Key/query dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Attend a batch of queries against keys and values. `queries` are `Lq × d`,
    /// `keys` are `Lk × d`, `values` are `Lk × dv`; the result is `Lq × dv`.
    pub fn attend(&self, queries: &[&[f32]], keys: &[&[f32]], values: &[&[f32]]) -> Vec<Vec<f32>> {
        let lq = queries.len();
        let lk = keys.len();
        assert_eq!(lk, values.len(), "one value per key");
        if lq == 0 || lk == 0 {
            return vec![Vec::new(); lq];
        }
        let dv = values[0].len();

        // Q (Lq × d).
        let mut qd = vec![0.0f32; lq * self.d];
        for (i, q) in queries.iter().enumerate() {
            qd[i * self.d..(i + 1) * self.d].copy_from_slice(q);
        }
        let q_lat = PaddedTileLattice::from_dense(lq, self.d, &qd, self.geom).unwrap();

        // Kᵀ (d × Lk): column j is key j.
        let mut ktd = vec![0.0f32; self.d * lk];
        for (j, k) in keys.iter().enumerate() {
            for (p, &val) in k.iter().enumerate() {
                ktd[p * lk + j] = val;
            }
        }
        let kt_lat = PaddedTileLattice::from_dense(self.d, lk, &ktd, self.geom).unwrap();

        // scores = Q · Kᵀ / √d, then row-softmax.
        let scale = 1.0 / (self.d as f32).sqrt();
        let raw = q_lat.matmul(&kt_lat).unwrap().to_dense();
        let mut weights = vec![0.0f32; lq * lk];
        for i in 0..lq {
            let row = &raw[i * lk..(i + 1) * lk];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * scale;
            let mut sum = 0.0f32;
            for j in 0..lk {
                let e = ((row[j] * scale) - max).exp();
                weights[i * lk + j] = e;
                sum += e;
            }
            for j in 0..lk {
                weights[i * lk + j] /= sum;
            }
        }
        let w_lat = PaddedTileLattice::from_dense(lq, lk, &weights, self.geom).unwrap();

        // V (Lk × dv).
        let mut vd = vec![0.0f32; lk * dv];
        for (j, v) in values.iter().enumerate() {
            vd[j * dv..(j + 1) * dv].copy_from_slice(v);
        }
        let v_lat = PaddedTileLattice::from_dense(lk, dv, &vd, self.geom).unwrap();

        // out = weights · V.
        let out = w_lat.matmul(&v_lat).unwrap().to_dense();
        out.chunks_exact(dv).map(|r| r.to_vec()).collect()
    }

    /// Attend a single query, returning its `dv`-dimensional output.
    pub fn attend_one(&self, query: &[f32], keys: &[&[f32]], values: &[&[f32]]) -> Vec<f32> {
        self.attend(&[query], keys, values)
            .pop()
            .unwrap_or_default()
    }
}
