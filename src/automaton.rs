//! `TensorAutomaton` — a finite-state machine whose execution is matrix multiply.
//!
//! A deterministic finite automaton (DFA) is usually run with a branch: "look at
//! the current state and the next symbol, jump to the next state." Branches are
//! exactly what a systolic accelerator is bad at. So represent the machine as
//! tensors instead:
//!
//! - the current state is a **one-hot row vector** `s` of width `Q` (number of
//!   states);
//! - each input symbol `a` has a **transition matrix** `T_a` of shape `Q × Q`,
//!   where `T_a[i, j] = 1` iff reading `a` in state `i` moves to state `j`;
//! - one step is the matmul `s' = s · T_a`, which lands one-hot on the next state.
//!
//! Running a string is a chain of these matmuls — no branches at all. A whole
//! batch of strings advances with `|alphabet|` masked matmuls per position,
//! independent of the batch size, which is what makes this a tensor-shaped way to
//! recognise strings. See [`TensorAutomaton::mod_k`] for a worked example: testing
//! divisibility by `k` purely through matrix multiplies.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;

/// A deterministic finite automaton executed as matrix multiplies.
#[derive(Clone)]
pub struct TensorAutomaton {
    num_states: usize,
    num_symbols: usize,
    start: usize,
    accept: Vec<bool>,
    geom: Geometry,
    /// `matrices[a]` is the `Q × Q` transition matrix for symbol `a`.
    matrices: Vec<PaddedTileLattice<f32>>,
}

impl TensorAutomaton {
    /// Build an automaton from an explicit transition table: `trans[state][symbol]`
    /// is the next state.
    pub fn new(start: usize, accept: Vec<bool>, trans: Vec<Vec<usize>>) -> Self {
        let num_states = trans.len();
        assert!(num_states > 0, "need at least one state");
        assert_eq!(accept.len(), num_states, "accept flag per state");
        let num_symbols = trans[0].len();
        let geom = Geometry::TPU_V;
        let matrices = (0..num_symbols)
            .map(|a| {
                let mut dense = vec![0.0f32; num_states * num_states];
                for (s, row) in trans.iter().enumerate() {
                    let next = row[a];
                    assert!(next < num_states, "transition out of range");
                    dense[s * num_states + next] = 1.0;
                }
                PaddedTileLattice::from_dense(num_states, num_states, &dense, geom).unwrap()
            })
            .collect();
        TensorAutomaton {
            num_states,
            num_symbols,
            start,
            accept,
            geom,
            matrices,
        }
    }

    /// The automaton over alphabet `{0, 1}` that accepts binary strings (most
    /// significant bit first) whose value is divisible by `k`. State `s` is the
    /// running value mod `k`; reading bit `b` goes to `(2s + b) mod k`.
    pub fn mod_k(k: usize) -> Self {
        assert!(k > 0, "modulus must be positive");
        let trans: Vec<Vec<usize>> = (0..k).map(|s| vec![(2 * s) % k, (2 * s + 1) % k]).collect();
        let accept = (0..k).map(|s| s == 0).collect();
        TensorAutomaton::new(0, accept, trans)
    }

    /// Number of states `Q`.
    #[inline]
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Alphabet size.
    #[inline]
    pub fn num_symbols(&self) -> usize {
        self.num_symbols
    }

    /// True if `state` is an accepting state.
    #[inline]
    pub fn is_accepting(&self, state: usize) -> bool {
        self.accept[state]
    }

    /// The transition matrix for `symbol` (a `Q × Q` lattice).
    #[inline]
    pub fn transition(&self, symbol: usize) -> &PaddedTileLattice<f32> {
        &self.matrices[symbol]
    }

    /// Index of the single set entry in a one-hot row.
    fn argmax_row(row: &[f32]) -> usize {
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, &v) in row.iter().enumerate() {
            if v > best.1 {
                best = (i, v);
            }
        }
        best.0
    }

    /// Run `input` from the start state and return the final state, advancing one
    /// matmul per symbol.
    pub fn run(&self, input: &[usize]) -> usize {
        let q = self.num_states;
        let mut state = vec![0.0f32; q];
        state[self.start] = 1.0;
        for &sym in input {
            let s = PaddedTileLattice::from_dense(1, q, &state, self.geom).unwrap();
            let next = s.matmul(&self.matrices[sym]).unwrap();
            state = next.to_dense();
        }
        TensorAutomaton::argmax_row(&state)
    }

    /// True if `input` is accepted.
    pub fn accepts(&self, input: &[usize]) -> bool {
        self.accept[self.run(input)]
    }

    /// Run a whole batch of strings, advancing every string at position `t` with
    /// `|alphabet|` masked matmuls — independent of how many strings there are.
    /// Returns the final state of each string.
    pub fn run_batch(&self, inputs: &[&[usize]]) -> Vec<usize> {
        let b = inputs.len();
        let q = self.num_states;
        if b == 0 {
            return Vec::new();
        }
        let max_len = inputs.iter().map(|s| s.len()).max().unwrap_or(0);

        // One-hot start state per string.
        let mut states = vec![0.0f32; b * q];
        for row in states.chunks_exact_mut(q) {
            row[self.start] = 1.0;
        }

        for t in 0..max_len {
            let mut next = vec![0.0f32; b * q];
            // Strings that have ended carry their state unchanged.
            for (i, input) in inputs.iter().enumerate() {
                if t >= input.len() {
                    next[i * q..(i + 1) * q].copy_from_slice(&states[i * q..(i + 1) * q]);
                }
            }
            // For each symbol, advance exactly the rows reading that symbol at t.
            for a in 0..self.num_symbols {
                let mut masked = vec![0.0f32; b * q];
                let mut any = false;
                for (i, input) in inputs.iter().enumerate() {
                    if t < input.len() && input[t] == a {
                        masked[i * q..(i + 1) * q].copy_from_slice(&states[i * q..(i + 1) * q]);
                        any = true;
                    }
                }
                if !any {
                    continue;
                }
                let m = PaddedTileLattice::from_dense(b, q, &masked, self.geom).unwrap();
                let advanced = m.matmul(&self.matrices[a]).unwrap().to_dense();
                for (n, adv) in next.iter_mut().zip(&advanced) {
                    *n += adv;
                }
            }
            states = next;
        }

        states
            .chunks_exact(q)
            .map(TensorAutomaton::argmax_row)
            .collect()
    }

    /// Accept/reject every string in a batch.
    pub fn batch_accepts(&self, inputs: &[&[usize]]) -> Vec<bool> {
        self.run_batch(inputs)
            .into_iter()
            .map(|s| self.accept[s])
            .collect()
    }
}

impl core::fmt::Debug for TensorAutomaton {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "TensorAutomaton {{ states: {}, symbols: {} }}",
            self.num_states, self.num_symbols
        )
    }
}
