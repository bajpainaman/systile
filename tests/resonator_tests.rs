//! Tests for the resonator-network factorization.

use systile::Resonator;

#[test]
fn compose_then_factorize_two_factors() {
    let res = Resonator::uniform(4096, 2, 40, 1);
    let truth = [11usize, 27];
    let c = res.compose(&truth);
    let r = res.factorize(&c, 200);
    assert_eq!(r.factors, truth.to_vec());
    assert!(r.verified);
}

#[test]
fn compose_then_factorize_three_factors() {
    let res = Resonator::uniform(8192, 3, 50, 2);
    let truth = [7usize, 42, 13];
    let c = res.compose(&truth);
    let r = res.factorize(&c, 200);
    assert_eq!(r.factors, truth.to_vec());
    assert!(r.verified);
}

#[test]
fn verify_accepts_truth_and_rejects_lie() {
    let res = Resonator::uniform(2048, 3, 30, 3);
    let truth = [1usize, 2, 3];
    let c = res.compose(&truth);
    assert!(res.verify(&c, &truth));
    assert!(!res.verify(&c, &[1, 2, 4]));
}

#[test]
fn factorize_is_robust_across_seeds() {
    // Restarts + exact verification should resolve every instance.
    for seed in [10u64, 20, 30, 40, 50] {
        let res = Resonator::uniform(4096, 3, 40, seed);
        let truth = [3usize, 17, 29];
        let c = res.compose(&truth);
        let r = res.factorize(&c, 200);
        assert!(r.verified, "seed {seed} failed to verify");
        assert_eq!(r.factors, truth.to_vec());
    }
}

#[test]
fn factors_and_dim_accessors() {
    let res = Resonator::uniform(1024, 4, 16, 1);
    assert_eq!(res.factors(), 4);
    assert_eq!(res.dim(), 1024);
    assert_eq!(res.codebook(0).len(), 16);
}

#[test]
fn single_factor_is_just_cleanup() {
    let res = Resonator::uniform(2048, 1, 64, 7);
    let c = res.compose(&[33]);
    let r = res.factorize(&c, 50);
    assert_eq!(r.factors, vec![33]);
    assert!(r.verified);
}

#[test]
fn different_truths_recover_independently() {
    let res = Resonator::uniform(4096, 3, 32, 9);
    for truth in [[0usize, 0, 0], [5, 10, 15], [31, 1, 16]] {
        let c = res.compose(&truth);
        let r = res.factorize(&c, 200);
        assert_eq!(r.factors, truth.to_vec(), "truth {truth:?}");
    }
}
