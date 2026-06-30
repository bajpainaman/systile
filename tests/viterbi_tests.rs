//! Tests for max-plus-matmul Viterbi decoding.

use systile::TensorViterbi;

/// The umbrella HMM: 0 = Sunny, 1 = Rainy; obs 0 = none, 1 = umbrella.
fn umbrella_hmm() -> TensorViterbi {
    let mut h = TensorViterbi::new(2, 2);
    h.set_initial(0, 0.6);
    h.set_initial(1, 0.4);
    h.set_transition(0, 0, 0.7);
    h.set_transition(0, 1, 0.3);
    h.set_transition(1, 0, 0.4);
    h.set_transition(1, 1, 0.6);
    h.set_emission(0, 0, 0.8);
    h.set_emission(0, 1, 0.2);
    h.set_emission(1, 0, 0.1);
    h.set_emission(1, 1, 0.9);
    h
}

/// Brute-force most-likely path for cross-checking small cases.
fn brute_force(h: &TensorViterbi, obs: &[usize], states: usize) -> Vec<usize> {
    fn rec(
        h: &TensorViterbi,
        obs: &[usize],
        states: usize,
        t: usize,
        path: &mut Vec<usize>,
        best: &mut (f32, Vec<usize>),
    ) {
        if t == obs.len() {
            let lp = h.path_log_prob(path, obs);
            if lp > best.0 {
                *best = (lp, path.clone());
            }
            return;
        }
        for s in 0..states {
            path.push(s);
            rec(h, obs, states, t + 1, path, best);
            path.pop();
        }
    }
    let mut best = (f32::NEG_INFINITY, Vec::new());
    rec(h, obs, states, 0, &mut Vec::new(), &mut best);
    best.1
}

#[test]
fn umbrella_sequence_decodes_to_rainy() {
    let h = umbrella_hmm();
    let (path, _) = h.decode(&[1, 1, 0, 1]);
    assert_eq!(path, vec![1, 1, 0, 1]);
}

#[test]
fn single_observation_picks_best_state() {
    let h = umbrella_hmm();
    // An umbrella alone is most likely Rainy.
    assert_eq!(h.decode(&[1]).0, vec![1]);
    // No umbrella alone is most likely Sunny.
    assert_eq!(h.decode(&[0]).0, vec![0]);
}

#[test]
fn empty_observations() {
    let h = umbrella_hmm();
    assert_eq!(h.decode(&[]).0, Vec::<usize>::new());
}

#[test]
fn path_length_matches_observations() {
    let h = umbrella_hmm();
    let obs = [1usize, 0, 1, 1, 0, 0, 1];
    assert_eq!(h.decode(&obs).0.len(), obs.len());
}

#[test]
fn matches_brute_force() {
    let h = umbrella_hmm();
    for obs in [
        vec![1usize, 1, 1],
        vec![0, 0, 0],
        vec![1, 0, 1, 0],
        vec![0, 1, 1, 0, 1],
    ] {
        assert_eq!(h.decode(&obs).0, brute_force(&h, &obs, 2), "obs {obs:?}");
    }
}
