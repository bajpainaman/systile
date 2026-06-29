//! Tests for the hyperdimensional algebra.

use systile::Hyper;

#[test]
fn atom_is_bipolar() {
    let a = Hyper::atom(1024, 1, 0);
    assert!(a.as_slice().iter().all(|&x| x == 1.0 || x == -1.0));
}

#[test]
fn atom_is_deterministic() {
    assert_eq!(Hyper::atom(512, 7, 3), Hyper::atom(512, 7, 3));
}

#[test]
fn different_ids_differ() {
    assert_ne!(Hyper::atom(512, 7, 3), Hyper::atom(512, 7, 4));
}

#[test]
fn different_seeds_differ() {
    assert_ne!(Hyper::atom(512, 1, 0), Hyper::atom(512, 2, 0));
}

#[test]
fn distinct_atoms_are_near_orthogonal() {
    let a = Hyper::atom(8192, 42, 0);
    let b = Hyper::atom(8192, 42, 1);
    // Random bipolar vectors of dim d have cosine ~ N(0, 1/d); |cos| << 0.1.
    assert!(a.cosine(&b).abs() < 0.1, "cosine was {}", a.cosine(&b));
}

#[test]
fn atom_is_unit_under_cosine_with_self() {
    let a = Hyper::atom(1024, 3, 9);
    assert!((a.cosine(&a) - 1.0).abs() < 1e-6);
}

#[test]
fn bind_is_self_inverse() {
    let a = Hyper::atom(1024, 1, 0);
    let b = Hyper::atom(1024, 1, 1);
    assert_eq!(a.bind(&b).bind(&b), a);
}

#[test]
fn bind_is_commutative() {
    let a = Hyper::atom(256, 1, 0);
    let b = Hyper::atom(256, 1, 1);
    assert_eq!(a.bind(&b), b.bind(&a));
}

#[test]
fn bind_is_dissimilar_to_operands() {
    let a = Hyper::atom(8192, 1, 0);
    let b = Hyper::atom(8192, 1, 1);
    let c = a.bind(&b);
    assert!(c.cosine(&a).abs() < 0.1);
    assert!(c.cosine(&b).abs() < 0.1);
}

#[test]
fn bundle_is_similar_to_members() {
    let a = Hyper::atom(8192, 1, 0);
    let b = Hyper::atom(8192, 1, 1);
    let mut bundle = Hyper::zeros(8192);
    bundle.bundle_into(&a);
    bundle.bundle_into(&b);
    // The superposition is positively correlated with each member.
    assert!(bundle.cosine(&a) > 0.5);
    assert!(bundle.cosine(&b) > 0.5);
}

#[test]
fn unbundle_reverses_bundle() {
    let a = Hyper::atom(512, 1, 0);
    let b = Hyper::atom(512, 1, 1);
    let mut bundle = a.clone();
    bundle.bundle_into(&b);
    bundle.unbundle(&b);
    assert_eq!(bundle, a);
}

#[test]
fn sign_returns_bipolar() {
    let v = Hyper::from_vec(vec![3.0, -2.0, 0.0, -0.5]);
    assert_eq!(v.sign().as_slice(), &[1.0, -1.0, 1.0, -1.0]);
}

#[test]
fn permute_then_inverse_is_identity() {
    let a = Hyper::atom(257, 1, 0);
    assert_eq!(a.permute(5).inverse_permute(5), a);
}

#[test]
fn permute_changes_the_vector() {
    let a = Hyper::atom(256, 1, 0);
    assert_ne!(a.permute(1), a);
}

#[test]
fn permute_preserves_norm() {
    let a = Hyper::atom(256, 1, 0);
    assert!((a.permute(7).norm() - a.norm()).abs() < 1e-6);
}

#[test]
fn dot_of_atom_with_self_is_dim() {
    let a = Hyper::atom(1024, 1, 0);
    assert_eq!(a.dot(&a), 1024.0);
}
