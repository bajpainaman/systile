//! Decode the most-likely weather sequence from "umbrella" observations with a
//! Viterbi pass of max-plus matmuls.
//!
//! Run with `cargo run --release --example viterbi_decode`.

use systile::prelude::*;

// Hidden states: 0 = Sunny, 1 = Rainy. Observation: 0 = no umbrella, 1 = umbrella.
fn main() {
    let mut hmm = TensorViterbi::new(2, 2);
    hmm.set_initial(0, 0.6);
    hmm.set_initial(1, 0.4);
    // Weather tends to persist.
    hmm.set_transition(0, 0, 0.7);
    hmm.set_transition(0, 1, 0.3);
    hmm.set_transition(1, 0, 0.4);
    hmm.set_transition(1, 1, 0.6);
    // People carry umbrellas mostly when it rains.
    hmm.set_emission(0, 0, 0.8);
    hmm.set_emission(0, 1, 0.2);
    hmm.set_emission(1, 0, 0.1);
    hmm.set_emission(1, 1, 0.9);

    // Observations: umbrella, umbrella, none, umbrella.
    let obs = [1usize, 1, 0, 1];
    let names = ["Sunny", "Rainy"];
    let obs_names = ["no-umbrella", "umbrella"];

    println!("{hmm:?}");
    print!("observations: ");
    for &o in &obs {
        print!("{} ", obs_names[o]);
    }
    println!();

    let (path, logp) = hmm.decode(&obs);
    print!("most-likely weather: ");
    for &s in &path {
        print!("{} ", names[s]);
    }
    println!("\nlog-probability: {logp:.3}");

    // Three umbrella days (with one gap) should be decoded as mostly Rainy.
    assert_eq!(path, vec![1, 1, 0, 1]);
    println!("\n✓ best path found by chaining max-plus matmuls + traceback.");
}
