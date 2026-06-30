//! `HoloSequence` — an ordered sequence packed into one hypervector by binding
//! each element to its position with a permutation.
//!
//! Bundling alone is a *set*: it forgets order. To remember position `i`, rotate
//! the element's hypervector by `i` before bundling — `memory = Σ ρ^i(atom(s_i))`.
//! Rotation is a permutation that makes the same symbol look different at each
//! position, so reading position `i` back is just the inverse rotation followed by
//! cleanup. Decoding the whole sequence is one batched matmul: every position's
//! un-rotated probe is cleaned up against the codebook at once.

use crate::codebook::Codebook;
use crate::hyper::Hyper;

/// An ordered sequence of symbol ids stored in one superposition vector.
#[derive(Clone)]
pub struct HoloSequence {
    codebook: Codebook,
    memory: Hyper,
    len: usize,
}

impl HoloSequence {
    /// Create an empty sequence over a `dim`-dimensional space with `alphabet`
    /// possible symbols.
    pub fn new(dim: usize, alphabet: usize, seed: u64) -> Self {
        let codebook = Codebook::new(dim, alphabet, seed);
        let memory = Hyper::zeros(dim);
        HoloSequence {
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

    /// Current sequence length.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the sequence is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The underlying superposition vector — the whole sequence in one tensor.
    #[inline]
    pub fn memory(&self) -> &Hyper {
        &self.memory
    }

    /// Append `id` to the end of the sequence, binding it to its position by a
    /// rotation of `len`.
    pub fn push(&mut self, id: usize) {
        let positioned = self.codebook.atom(id).permute(self.len);
        self.memory.bundle_into(&positioned);
        self.len += 1;
    }

    /// The un-rotated probe for position `pos`, ready for cleanup.
    fn probe(&self, pos: usize) -> Hyper {
        self.memory.inverse_permute(pos)
    }

    /// Read the symbol at `pos`, returning `(symbol_id, score)`.
    pub fn get(&self, pos: usize) -> (usize, f32) {
        self.codebook.cleanup(&self.probe(pos))
    }

    /// Decode the whole sequence with a single batched matmul over all positions.
    pub fn decode(&self) -> Vec<usize> {
        let probes: Vec<Hyper> = (0..self.len).map(|p| self.probe(p)).collect();
        self.codebook
            .cleanup_batch(&probes)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }
}

impl core::fmt::Debug for HoloSequence {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "HoloSequence {{ dim: {}, len: {} }}",
            self.dim(),
            self.len
        )
    }
}
