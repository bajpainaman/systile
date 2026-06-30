//! `HoloClassifier` — classification where training is addition and inference is
//! a matmul.
//!
//! This is the hyperdimensional-computing classifier (Kanerva; Rahimi et al.). It
//! has no weights to descend, no epochs, no backprop:
//!
//! - **Encode** a sample into a hypervector by binding each feature's *position*
//!   to a *level* hypervector for its quantised value, then bundling.
//! - **Train** by bundling every training sample of a class into that class's
//!   prototype vector — so "fitting" the model is literally vector addition.
//! - **Infer** by comparing a sample to every class prototype at once: stack the
//!   prototypes into a `dim × classes` matrix and the answer is `argmax` of a
//!   single `(batch × dim) · (dim × classes)` matmul.
//!
//! The model *is* the data structure (a handful of prototype vectors), and
//! inference *is* the codebook-cleanup matmul from [`crate::codebook`].

use crate::geometry::Geometry;
use crate::hyper::Hyper;
use crate::lattice::PaddedTileLattice;

#[inline]
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A train-by-bundling, infer-by-matmul hyperdimensional classifier.
#[derive(Clone)]
pub struct HoloClassifier {
    dim: usize,
    n_features: usize,
    n_levels: usize,
    seed: u64,
    geom: Geometry,
    levels: Vec<Hyper>,
    prototypes: Vec<Hyper>,
    counts: Vec<usize>,
}

impl HoloClassifier {
    /// Create a classifier over `n_features` features, each quantised to
    /// `n_levels` levels, with `n_classes` output classes.
    pub fn new(
        dim: usize,
        n_features: usize,
        n_levels: usize,
        n_classes: usize,
        seed: u64,
    ) -> Self {
        assert!(n_levels >= 1, "need at least one level");
        let levels = HoloClassifier::build_levels(dim, n_levels, seed ^ 0x1E_5E15);
        HoloClassifier {
            dim,
            n_features,
            n_levels,
            seed,
            geom: Geometry::TPU_V,
            levels,
            prototypes: vec![Hyper::zeros(dim); n_classes],
            counts: vec![0; n_classes],
        }
    }

    /// Build `n_levels` "scalar" hypervectors where adjacent levels are similar and
    /// the extremes are near-orthogonal, by progressively flipping bits of `L0`.
    fn build_levels(dim: usize, n_levels: usize, seed: u64) -> Vec<Hyper> {
        let base = Hyper::atom(dim, seed, 0);
        if n_levels == 1 {
            return vec![base];
        }
        // A pseudo-random order in which to flip coordinates.
        let mut order: Vec<usize> = (0..dim).collect();
        let mut state = seed ^ 0xABCD;
        for i in (1..dim).rev() {
            state = splitmix64(state);
            let j = (state as usize) % (i + 1);
            order.swap(i, j);
        }
        let total_flips = dim / 2; // extreme level is ~orthogonal to L0
        let base_vec = base.as_slice().to_vec();
        (0..n_levels)
            .map(|i| {
                let nflip = total_flips * i / (n_levels - 1);
                let mut v = base_vec.clone();
                for &idx in order.iter().take(nflip) {
                    v[idx] = -v[idx];
                }
                Hyper::from_vec(v)
            })
            .collect()
    }

    /// Number of classes.
    #[inline]
    pub fn n_classes(&self) -> usize {
        self.prototypes.len()
    }

    /// Number of training samples seen for `class`.
    #[inline]
    pub fn class_count(&self, class: usize) -> usize {
        self.counts[class]
    }

    /// The position hypervector for feature `f`.
    fn position(&self, f: usize) -> Hyper {
        Hyper::atom(self.dim, self.seed ^ 0x9051, f as u64)
    }

    /// Encode a sample (one quantised level index per feature) into a bipolar
    /// hypervector.
    pub fn encode(&self, features: &[usize]) -> Hyper {
        assert_eq!(features.len(), self.n_features, "one level per feature");
        let mut acc = Hyper::zeros(self.dim);
        for (f, &level) in features.iter().enumerate() {
            debug_assert!(level < self.n_levels, "level out of range");
            acc.bundle_into(&self.position(f).bind(&self.levels[level]));
        }
        acc.sign()
    }

    /// Train on one labelled sample by bundling it into its class prototype.
    pub fn train(&mut self, features: &[usize], class: usize) {
        let encoded = self.encode(features);
        self.prototypes[class].bundle_into(&encoded);
        self.counts[class] += 1;
    }

    /// Stack the class prototypes into a `dim × classes` lattice for cleanup.
    fn prototype_matrix(&self) -> PaddedTileLattice<f32> {
        let c = self.n_classes();
        let mut dense = vec![0.0f32; self.dim * c];
        for (j, proto) in self.prototypes.iter().enumerate() {
            for (p, &value) in proto.as_slice().iter().enumerate() {
                dense[p * c + j] = value;
            }
        }
        PaddedTileLattice::from_dense(self.dim, c, &dense, self.geom).unwrap()
    }

    /// Classify a batch of samples with one matmul against the prototype matrix.
    pub fn classify_batch(&self, samples: &[&[usize]]) -> Vec<usize> {
        let b = samples.len();
        if b == 0 {
            return Vec::new();
        }
        let c = self.n_classes();
        let mut qdense = vec![0.0f32; b * self.dim];
        for (i, s) in samples.iter().enumerate() {
            let enc = self.encode(s);
            qdense[i * self.dim..(i + 1) * self.dim].copy_from_slice(enc.as_slice());
        }
        let q = PaddedTileLattice::from_dense(b, self.dim, &qdense, self.geom).unwrap();
        let scores = q.matmul(&self.prototype_matrix()).unwrap().to_dense();
        scores
            .chunks_exact(c)
            .map(|row| {
                let mut best = (0usize, f32::NEG_INFINITY);
                for (j, &s) in row.iter().enumerate() {
                    if s > best.1 {
                        best = (j, s);
                    }
                }
                best.0
            })
            .collect()
    }

    /// Classify a single sample.
    pub fn classify(&self, features: &[usize]) -> usize {
        self.classify_batch(&[features])[0]
    }
}

impl core::fmt::Debug for HoloClassifier {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "HoloClassifier {{ dim: {}, features: {}, levels: {}, classes: {} }}",
            self.dim,
            self.n_features,
            self.n_levels,
            self.n_classes()
        )
    }
}
