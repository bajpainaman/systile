//! Resonator networks: factoring a bound product back into its unknown symbols
//! with iterated matmuls.
//!
//! Binding is easy to do and, given the keys, easy to undo. But suppose you are
//! handed a composite `s = x ⊛ y ⊛ z` and you do *not* know any of the three
//! factors — only that each is some atom from a known codebook. Brute force is
//! `M^F` combinations. A **resonator network** (Frady, Kent, Olshausen & Sommer,
//! 2020) solves it by holding a superposed estimate of each factor and letting them
//! "resonate": each estimate is repeatedly cleaned up against its codebook while
//! the others are unbound from `s`. In a handful of iterations the estimates snap
//! onto the true atoms.
//!
//! Every step is dense matmul against a codebook ([`Codebook::scores_batch`] then
//! [`Codebook::superpose`]), so the whole combinatorial search runs as a short
//! sequence of MXU operations — a search algorithm expressed entirely as matmuls.

use crate::codebook::Codebook;
use crate::hyper::Hyper;

/// The outcome of a factorization attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct Factorization {
    /// The recovered symbol id for each factor, in codebook order.
    pub factors: Vec<usize>,
    /// How many iterations the winning attempt ran.
    pub iterations: usize,
    /// Whether the winning attempt reached a stable fixed point (vs. the cap).
    pub converged: bool,
    /// Whether recomposing `factors` reproduces the original composite exactly.
    /// When `true`, the answer is provably correct; when `false`, it is a guess.
    pub verified: bool,
    /// How many restarts it took (0 means the first attempt succeeded).
    pub restarts: usize,
}

/// A resonator network over `F` codebooks, one per factor of a bound product.
#[derive(Clone)]
pub struct Resonator {
    codebooks: Vec<Codebook>,
    dim: usize,
}

impl Resonator {
    /// Build a resonator from one codebook per factor. All must share a dimension.
    pub fn new(codebooks: Vec<Codebook>) -> Self {
        assert!(!codebooks.is_empty(), "need at least one factor");
        let dim = codebooks[0].dim();
        assert!(
            codebooks.iter().all(|c| c.dim() == dim),
            "all factor codebooks must share the same dimension"
        );
        Resonator { codebooks, dim }
    }

    /// Build `factors` independent codebooks of `count` symbols each, namespaced
    /// from `seed`. A composite is then one atom from each.
    pub fn uniform(dim: usize, factors: usize, count: usize, seed: u64) -> Self {
        let codebooks = (0..factors)
            .map(|f| {
                Codebook::new(
                    dim,
                    count,
                    seed.wrapping_add(f as u64).wrapping_mul(0x9E37_79B9),
                )
            })
            .collect();
        Resonator::new(codebooks)
    }

    /// Number of factors.
    #[inline]
    pub fn factors(&self) -> usize {
        self.codebooks.len()
    }

    /// Hypervector dimensionality.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Borrow factor `f`'s codebook.
    #[inline]
    pub fn codebook(&self, f: usize) -> &Codebook {
        &self.codebooks[f]
    }

    /// Bind one atom from each factor into a composite — the value you would later
    /// hand back to [`Resonator::factorize`].
    pub fn compose(&self, ids: &[usize]) -> Hyper {
        assert_eq!(ids.len(), self.factors(), "one id per factor");
        let mut acc = self.codebooks[0].atom(ids[0]);
        for (f, &id) in ids.iter().enumerate().skip(1) {
            acc = acc.bind(&self.codebooks[f].atom(id));
        }
        acc
    }

    /// Project a hypervector onto factor `f`'s codebook: `X (Xᵀ v)`, kept bipolar.
    fn project(&self, f: usize, v: &Hyper) -> Hyper {
        let scores = self.codebooks[f].scores_batch(core::slice::from_ref(v));
        self.codebooks[f].superpose(&scores[0]).sign()
    }

    /// The initial estimate for a factor on restart `r`. Restart 0 uses the broad
    /// "every atom at once" superposition; later restarts use independent random
    /// bipolar vectors to escape spurious fixed points.
    fn initial_estimate(&self, f: usize, restart: usize) -> Hyper {
        if restart == 0 {
            let ones = vec![1.0f32; self.codebooks[f].len()];
            self.codebooks[f].superpose(&ones).sign()
        } else {
            Hyper::atom(self.dim, 0xC0DE_F00D ^ restart as u64, f as u64)
        }
    }

    /// True if recomposing `ids` reproduces `composite` exactly — a free, exact
    /// check, because binding the correct atoms returns the original product.
    pub fn verify(&self, composite: &Hyper, ids: &[usize]) -> bool {
        ids.len() == self.factors() && self.compose(ids) == *composite
    }

    /// Run the resonator dynamics once from a given restart's initialization.
    fn run_once(&self, composite: &Hyper, max_iters: usize, restart: usize) -> (Vec<usize>, usize) {
        let f = self.factors();
        let mut est: Vec<Hyper> = (0..f).map(|i| self.initial_estimate(i, restart)).collect();
        let mut prev_ids: Vec<usize> = vec![usize::MAX; f];

        for iter in 1..=max_iters {
            // Synchronous (Jacobi) update: every new estimate uses the old others.
            let mut next = Vec::with_capacity(f);
            for target in 0..f {
                let mut unbound = composite.clone();
                for (other, e) in est.iter().enumerate() {
                    if other != target {
                        unbound = unbound.bind(e);
                    }
                }
                next.push(self.project(target, &unbound));
            }
            est = next;

            let ids: Vec<usize> = (0..f)
                .map(|i| self.codebooks[i].cleanup(&est[i]).0)
                .collect();
            if ids == prev_ids {
                return (ids, iter);
            }
            prev_ids = ids;
        }
        (prev_ids, max_iters)
    }

    /// Recover the factor ids of a composite `s = a ⊛ b ⊛ ...` without knowing any
    /// of them. Uses up to 25 restarts and stops as soon as a result *verifies*.
    pub fn factorize(&self, composite: &Hyper, max_iters: usize) -> Factorization {
        self.factorize_with(composite, max_iters, 25)
    }

    /// Like [`Resonator::factorize`] but with an explicit restart budget. Each
    /// restart re-seeds the estimates; the first verified solution wins.
    pub fn factorize_with(
        &self,
        composite: &Hyper,
        max_iters: usize,
        max_restarts: usize,
    ) -> Factorization {
        let mut fallback: Option<(Vec<usize>, usize, usize)> = None;
        for restart in 0..max_restarts.max(1) {
            let (ids, iters) = self.run_once(composite, max_iters, restart);
            let converged = iters < max_iters;
            if self.verify(composite, &ids) {
                return Factorization {
                    factors: ids,
                    iterations: iters,
                    converged,
                    verified: true,
                    restarts: restart,
                };
            }
            if fallback.is_none() {
                fallback = Some((ids, iters, restart));
            }
        }
        let (ids, iters, restart) = fallback.expect("at least one attempt runs");
        Factorization {
            factors: ids,
            iterations: iters,
            converged: false,
            verified: false,
            restarts: restart,
        }
    }
}

impl core::fmt::Debug for Resonator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Resonator {{ factors: {}, dim: {}, vocab: {} }}",
            self.factors(),
            self.dim,
            self.codebooks.first().map(|c| c.len()).unwrap_or(0)
        )
    }
}
