//! Tests for power-iteration PageRank.

use systile::TensorPageRank;

fn approx_sum_one(r: &[f32]) -> bool {
    (r.iter().sum::<f32>() - 1.0).abs() < 1e-4
}

#[test]
fn ranks_sum_to_one() {
    let mut pr = TensorPageRank::new(4);
    pr.add_link(0, 1);
    pr.add_link(1, 2);
    pr.add_link(2, 3);
    pr.add_link(3, 0);
    assert!(approx_sum_one(&pr.rank_default()));
}

#[test]
fn symmetric_cycle_is_uniform() {
    // A pure cycle is vertex-transitive: every node gets equal rank.
    let mut pr = TensorPageRank::new(5);
    for i in 0..5 {
        pr.add_link(i, (i + 1) % 5);
    }
    let r = pr.rank_default();
    for &v in &r {
        assert!((v - 0.2).abs() < 1e-3, "expected uniform, got {v}");
    }
}

#[test]
fn hub_ranks_highest() {
    // Everyone points at node 0.
    let mut pr = TensorPageRank::new(4);
    pr.add_link(1, 0);
    pr.add_link(2, 0);
    pr.add_link(3, 0);
    let r = pr.rank_default();
    let max_idx = (0..4)
        .max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap())
        .unwrap();
    assert_eq!(max_idx, 0);
}

#[test]
fn dangling_node_conserves_mass() {
    // Node 2 has no out-links; ranks must still sum to one.
    let mut pr = TensorPageRank::new(3);
    pr.add_link(0, 1);
    pr.add_link(1, 0);
    let r = pr.rank_default();
    assert!(approx_sum_one(&r));
}

#[test]
fn empty_graph_is_empty() {
    let pr = TensorPageRank::new(0);
    assert!(pr.rank_default().is_empty());
}

#[test]
fn single_node_has_rank_one() {
    let pr = TensorPageRank::new(1);
    let r = pr.rank_default();
    assert!((r[0] - 1.0).abs() < 1e-5);
}

#[test]
fn higher_damping_sharpens_distribution() {
    let mut pr = TensorPageRank::new(4);
    pr.add_link(1, 0);
    pr.add_link(2, 0);
    pr.add_link(3, 0);
    let low = pr.rank(0.5, 100, 1e-7);
    let high = pr.rank(0.95, 100, 1e-7);
    // With stronger damping, the hub's share grows.
    assert!(high[0] > low[0]);
}
