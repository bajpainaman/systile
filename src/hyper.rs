//! Hyperdimensional algebra: the primitive operations of a Vector Symbolic
//! Architecture (VSA), in the bipolar Multiply-Add-Permute (MAP) flavour.
//!
//! A *hypervector* is a long vector — thousands of dimensions — drawn so that two
//! independently generated ones are almost orthogonal. Three operations turn that
//! into an algebra you can build data structures out of:
//!
//! - **bind** (`⊛`, elementwise product): combines two hypervectors into a third
//!   that is dissimilar to both. Bipolar bind is its own inverse, because
//!   `x ⊛ x = 1`. This is how you attach a value to a key.
//! - **bundle** (`+`, sum): superposes many hypervectors into one that is *similar*
//!   to each of them. This is how a whole set lives in one vector.
//! - **permute** (`ρ`, a fixed rotation): protects a hypervector's role, e.g. to
//!   order a sequence.
//!
//! Retrieval is **similarity** — a dot product. Resolving a noisy hypervector
//! against a whole codebook of clean ones is therefore a matrix multiply, which is
//! the entire reason this maps onto a systolic matrix unit. See [`crate::holo`].

/// A hypervector. Atomic symbols are bipolar (`±1`); bundles are real-valued
/// because superposition sums integers.
#[derive(Clone, PartialEq)]
pub struct Hyper(Vec<f32>);

/// SplitMix64: a tiny, fast, well-distributed integer hash used to draw
/// deterministic pseudo-random hypervectors without any external dependency.
#[inline]
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Hyper {
    /// Wrap a raw coordinate vector.
    #[inline]
    pub fn from_vec(data: Vec<f32>) -> Self {
        Hyper(data)
    }

    /// Borrow the raw coordinates.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Consume into the raw coordinates.
    #[inline]
    pub fn into_vec(self) -> Vec<f32> {
        self.0
    }

    /// Dimensionality of the hypervector.
    #[inline]
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// A hypervector of all zeros, the identity element for [`Hyper::bundle_into`].
    pub fn zeros(dim: usize) -> Self {
        Hyper(vec![0.0; dim])
    }

    /// Draw a deterministic bipolar (`±1`) atomic hypervector.
    ///
    /// `seed` namespaces an independent codebook; `id` selects a symbol within it.
    /// The same `(dim, seed, id)` always yields the same hypervector, so no runtime
    /// randomness or stored table is needed to regenerate a symbol.
    pub fn atom(dim: usize, seed: u64, id: u64) -> Self {
        let mut v = Vec::with_capacity(dim);
        let base = splitmix64(seed ^ id.wrapping_mul(0xD1B5_4A32_D192_ED03));
        for p in 0..dim {
            let h = splitmix64(base.wrapping_add(p as u64));
            // Top bit gives a balanced ±1.
            v.push(if h >> 63 == 1 { 1.0 } else { -1.0 });
        }
        Hyper(v)
    }

    /// Bind two hypervectors with the elementwise product (`⊛`).
    ///
    /// Binding is associative, commutative, and — for bipolar operands — its own
    /// inverse: `a.bind(b).bind(b) == a`.
    pub fn bind(&self, other: &Hyper) -> Hyper {
        debug_assert_eq!(self.dim(), other.dim());
        Hyper(self.0.iter().zip(&other.0).map(|(a, b)| a * b).collect())
    }

    /// Bind in place.
    pub fn bind_assign(&mut self, other: &Hyper) {
        debug_assert_eq!(self.dim(), other.dim());
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a *= b;
        }
    }

    /// Add `other` into `self`, superposing it (the bundle accumulation step).
    pub fn bundle_into(&mut self, other: &Hyper) {
        debug_assert_eq!(self.dim(), other.dim());
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a += b;
        }
    }

    /// Subtract `other` from `self`, removing one term from a bundle.
    pub fn unbundle(&mut self, other: &Hyper) {
        debug_assert_eq!(self.dim(), other.dim());
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a -= b;
        }
    }

    /// Collapse a real-valued hypervector back to bipolar by taking the sign.
    /// Ties (exact zero) map to `+1`.
    pub fn sign(&self) -> Hyper {
        Hyper(
            self.0
                .iter()
                .map(|&x| if x >= 0.0 { 1.0 } else { -1.0 })
                .collect(),
        )
    }

    /// Cyclically rotate the coordinates right by `shift`, the permutation `ρ`.
    pub fn permute(&self, shift: usize) -> Hyper {
        let n = self.0.len();
        if n == 0 {
            return self.clone();
        }
        let s = shift % n;
        let mut out = vec![0.0; n];
        for (i, &x) in self.0.iter().enumerate() {
            out[(i + s) % n] = x;
        }
        Hyper(out)
    }

    /// Undo [`Hyper::permute`].
    pub fn inverse_permute(&self, shift: usize) -> Hyper {
        let n = self.0.len();
        if n == 0 {
            return self.clone();
        }
        self.permute(n - (shift % n))
    }

    /// Raw dot product with another hypervector — the unnormalised similarity.
    pub fn dot(&self, other: &Hyper) -> f32 {
        debug_assert_eq!(self.dim(), other.dim());
        self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum()
    }

    /// Euclidean norm.
    pub fn norm(&self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Cosine similarity in `[-1, 1]`; two random atoms score near `0`.
    pub fn cosine(&self, other: &Hyper) -> f32 {
        let denom = self.norm() * other.norm();
        if denom == 0.0 {
            0.0
        } else {
            self.dot(other) / denom
        }
    }
}

impl core::fmt::Debug for Hyper {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Hyper(dim={}, norm={:.2})", self.dim(), self.norm())
    }
}
