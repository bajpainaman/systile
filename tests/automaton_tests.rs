//! Tests for the matmul-executed finite automaton.

use systile::TensorAutomaton;

/// Big-endian bits of `n`.
fn bits(n: u32) -> Vec<usize> {
    if n == 0 {
        return vec![0];
    }
    let width = 32 - n.leading_zeros();
    (0..width).rev().map(|i| ((n >> i) & 1) as usize).collect()
}

#[test]
fn mod_k_shape() {
    let dfa = TensorAutomaton::mod_k(3);
    assert_eq!(dfa.num_states(), 3);
    assert_eq!(dfa.num_symbols(), 2);
    assert!(dfa.is_accepting(0));
    assert!(!dfa.is_accepting(1));
}

#[test]
fn divisibility_by_three() {
    let dfa = TensorAutomaton::mod_k(3);
    for n in 0..50u32 {
        assert_eq!(dfa.accepts(&bits(n)), n % 3 == 0, "n={n}");
    }
}

#[test]
fn divisibility_by_seven() {
    let dfa = TensorAutomaton::mod_k(7);
    for n in 0..100u32 {
        assert_eq!(dfa.accepts(&bits(n)), n % 7 == 0, "n={n}");
    }
}

#[test]
fn final_state_is_the_residue() {
    let dfa = TensorAutomaton::mod_k(5);
    for n in 0..40u32 {
        assert_eq!(dfa.run(&bits(n)), (n % 5) as usize, "n={n}");
    }
}

#[test]
fn batch_matches_individual() {
    let dfa = TensorAutomaton::mod_k(4);
    let inputs: Vec<Vec<usize>> = (0..30u32).map(bits).collect();
    let refs: Vec<&[usize]> = inputs.iter().map(|v| v.as_slice()).collect();
    let batch = dfa.batch_accepts(&refs);
    for (n, &ok) in batch.iter().enumerate() {
        assert_eq!(ok, dfa.accepts(&bits(n as u32)), "n={n}");
    }
}

#[test]
fn batch_handles_varied_lengths() {
    // Strings of different lengths must still each end in the right state.
    let dfa = TensorAutomaton::mod_k(3);
    let inputs: Vec<Vec<usize>> = [0u32, 3, 5, 9, 16, 21].iter().map(|&n| bits(n)).collect();
    let refs: Vec<&[usize]> = inputs.iter().map(|v| v.as_slice()).collect();
    let batch = dfa.batch_accepts(&refs);
    let expected: Vec<bool> = [0u32, 3, 5, 9, 16, 21]
        .iter()
        .map(|&n| n % 3 == 0)
        .collect();
    assert_eq!(batch, expected);
}

#[test]
fn custom_two_state_parity_automaton() {
    // States: 0 = even number of 1s (accept), 1 = odd. Symbol 0 keeps, 1 flips.
    let dfa = TensorAutomaton::new(0, vec![true, false], vec![vec![0, 1], vec![1, 0]]);
    assert!(dfa.accepts(&[1, 1])); // two ones -> even
    assert!(!dfa.accepts(&[1, 0, 1, 1])); // three ones -> odd
    assert!(dfa.accepts(&[0, 0, 0])); // zero ones -> even
}

#[test]
fn empty_batch_is_empty() {
    let dfa = TensorAutomaton::mod_k(3);
    assert!(dfa.batch_accepts(&[]).is_empty());
}
