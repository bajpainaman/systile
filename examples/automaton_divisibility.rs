//! Test divisibility by running a finite automaton as matrix multiplies.
//!
//! The mod-k automaton accepts a binary string (MSB first) iff its value is
//! divisible by k. Running it is a chain of one-hot · transition-matrix matmuls,
//! and a whole batch of numbers advances with just two masked matmuls per bit.
//!
//! Run with `cargo run --release --example automaton_divisibility`.

use systile::prelude::*;

/// Big-endian bits of `n` (most significant first).
fn bits(n: u32) -> Vec<usize> {
    if n == 0 {
        return vec![0];
    }
    let width = 32 - n.leading_zeros();
    (0..width).rev().map(|i| ((n >> i) & 1) as usize).collect()
}

fn main() {
    let k = 3;
    let dfa = TensorAutomaton::mod_k(k);
    println!("{dfa:?} — accepts binary strings divisible by {k}\n");

    // Batch-check 0..=20 with two masked matmuls per bit position.
    let inputs: Vec<Vec<usize>> = (0..=20u32).map(bits).collect();
    let refs: Vec<&[usize]> = inputs.iter().map(|v| v.as_slice()).collect();
    let verdicts = dfa.batch_accepts(&refs);

    print!("divisible by {k}:  ");
    for (n, &ok) in verdicts.iter().enumerate() {
        if ok {
            print!("{n} ");
        }
    }
    println!();

    // Cross-check against real arithmetic.
    for (n, &ok) in verdicts.iter().enumerate() {
        assert_eq!(ok, n % k == 0, "n={n}");
    }
    println!("\n✓ every verdict matches n % {k} == 0 — divisibility decided by matmul.");
}
