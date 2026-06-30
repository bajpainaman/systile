//! Factor a bound product back into its unknown symbols with a resonator network.
//!
//! Run with `cargo run --release --example resonator_factor`.

use systile::prelude::*;

fn main() {
    // Three factors, each an unknown symbol from a 50-word codebook. Brute force
    // would be 50^3 = 125_000 combinations.
    let res = Resonator::uniform(2048, 3, 50, 0xABCDEF);

    let truth = [7usize, 42, 13];
    let composite = res.compose(&truth); // s = a7 ⊛ b42 ⊛ c13, a single vector

    println!("composite hides factors {truth:?} in {} dims", res.dim());
    println!(
        "brute-force search space: 50^3 = {} combinations\n",
        50usize.pow(3)
    );

    let result = res.factorize(&composite, 100);

    println!("recovered: {:?}", result.factors);
    println!(
        "verified: {} | converged: {} | {} iters | {} restart(s)",
        result.verified, result.converged, result.iterations, result.restarts
    );
    assert_eq!(result.factors, truth.to_vec());
    assert!(result.verified);
    println!("\n✓ resolved by iterated matmul, not by enumerating the product space");
}
