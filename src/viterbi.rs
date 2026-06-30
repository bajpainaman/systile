//! `TensorViterbi` — most-likely hidden-state decoding as max-plus matmul.
//!
//! A hidden Markov model assigns a probability to a path of hidden states given a
//! sequence of observations. In log space, multiplying probabilities becomes adding
//! log-probabilities and maximising over paths becomes a `max`, so the Viterbi
//! recursion
//!
//! ```text
//! score_t[s] = max_p ( score_{t-1}[p] + logT[p, s] ) + logE[s, obs_t]
//! ```
//!
//! is a matrix multiply over the **max-plus semiring** ([`crate::semiring::MaxPlus`]):
//! `score_{t-1} ⊗ logT`. Each timestep is one such matmul (plus a back-pointer for
//! the final traceback), so decoding a sequence is a chain of max-plus GEMVs.

use crate::geometry::Geometry;
use crate::lattice::PaddedTileLattice;
use crate::semiring::{semiring_matmul, MaxPlus};

/// A hidden Markov model decoded with the Viterbi algorithm.
#[derive(Clone)]
pub struct TensorViterbi {
    states: usize,
    observations: usize,
    geom: Geometry,
    log_init: Vec<f32>,
    log_trans: Vec<f32>,
    log_emit: Vec<f32>,
}

impl TensorViterbi {
    /// Create a model with `states` hidden states and an alphabet of `observations`
    /// symbols. All probabilities start at zero (`−∞` in log space); set the ones
    /// you need.
    pub fn new(states: usize, observations: usize) -> Self {
        TensorViterbi {
            states,
            observations,
            geom: Geometry::TPU_V,
            log_init: vec![f32::NEG_INFINITY; states],
            log_trans: vec![f32::NEG_INFINITY; states * states],
            log_emit: vec![f32::NEG_INFINITY; states * observations],
        }
    }

    /// Number of hidden states.
    #[inline]
    pub fn states(&self) -> usize {
        self.states
    }

    /// Set the initial probability of `state`.
    pub fn set_initial(&mut self, state: usize, prob: f32) {
        self.log_init[state] = prob.ln();
    }

    /// Set the transition probability `from → to`.
    pub fn set_transition(&mut self, from: usize, to: usize, prob: f32) {
        self.log_trans[from * self.states + to] = prob.ln();
    }

    /// Set the emission probability of `symbol` in `state`.
    pub fn set_emission(&mut self, state: usize, symbol: usize, prob: f32) {
        self.log_emit[state * self.observations + symbol] = prob.ln();
    }

    /// The log-probability that the model emits `obs` while following the hidden
    /// state path `states`. Useful for scoring candidate paths.
    pub fn path_log_prob(&self, states: &[usize], obs: &[usize]) -> f32 {
        if states.is_empty() || states.len() != obs.len() {
            return f32::NEG_INFINITY;
        }
        let s = self.states;
        let mut lp =
            self.log_init[states[0]] + self.log_emit[states[0] * self.observations + obs[0]];
        for t in 1..obs.len() {
            lp += self.log_trans[states[t - 1] * s + states[t]]
                + self.log_emit[states[t] * self.observations + obs[t]];
        }
        lp
    }

    /// Decode the most-likely hidden-state path for `obs`, returning the path and
    /// its total log-probability.
    pub fn decode(&self, obs: &[usize]) -> (Vec<usize>, f32) {
        let s = self.states;
        if obs.is_empty() {
            return (Vec::new(), 0.0);
        }
        let trans = PaddedTileLattice::from_dense(s, s, &self.log_trans, self.geom).unwrap();

        // t = 0: initial + emission.
        let mut score: Vec<f32> = (0..s)
            .map(|st| self.log_init[st] + self.log_emit[st * self.observations + obs[0]])
            .collect();
        let mut back: Vec<Vec<usize>> = Vec::with_capacity(obs.len());
        back.push(vec![0; s]);

        for &symbol in &obs[1..] {
            // Max-plus matmul: best incoming score for each state.
            let sv = PaddedTileLattice::from_dense(1, s, &score, self.geom).unwrap();
            let incoming = semiring_matmul::<MaxPlus>(&sv, &trans).unwrap().to_dense();

            // Back-pointers: which predecessor achieved each max.
            let mut ptr = vec![0usize; s];
            for (to, p) in ptr.iter_mut().enumerate() {
                let mut best = (0usize, f32::NEG_INFINITY);
                for (from, &sc) in score.iter().enumerate() {
                    let v = sc + self.log_trans[from * s + to];
                    if v > best.1 {
                        best = (from, v);
                    }
                }
                *p = best.0;
            }
            back.push(ptr);

            score = (0..s)
                .map(|st| incoming[st] + self.log_emit[st * self.observations + symbol])
                .collect();
        }

        // Trace back from the best final state.
        let mut best_last = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (st, &v) in score.iter().enumerate() {
            if v > best_score {
                best_score = v;
                best_last = st;
            }
        }
        let mut path = vec![0usize; obs.len()];
        path[obs.len() - 1] = best_last;
        for t in (1..obs.len()).rev() {
            path[t - 1] = back[t][path[t]];
        }
        (path, best_score)
    }
}

impl core::fmt::Debug for TensorViterbi {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "TensorViterbi {{ states: {}, observations: {} }}",
            self.states, self.observations
        )
    }
}
