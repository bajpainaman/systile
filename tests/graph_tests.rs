//! Tests for the semiring-matmul graph engine.

use systile::semiring::{semiring_matmul, Boolean, Tropical};
use systile::{Geometry, PaddedTileLattice, TensorGraph};

fn line_graph() -> TensorGraph {
    // 0 -> 1 -> 2 -> 3, unit weights.
    let mut g = TensorGraph::new(4);
    g.add_edge_unweighted(0, 1);
    g.add_edge_unweighted(1, 2);
    g.add_edge_unweighted(2, 3);
    g
}

#[test]
fn node_and_edge_counts() {
    let g = line_graph();
    assert_eq!(g.nodes(), 4);
    assert_eq!(g.edge_count(), 3);
}

#[test]
fn reachability_follows_edges() {
    let g = line_graph();
    assert!(g.reachable(0, 3));
    assert!(g.reachable(1, 3));
    assert!(!g.reachable(3, 0));
}

#[test]
fn reachability_is_reflexive() {
    let g = line_graph();
    for i in 0..g.nodes() {
        assert!(g.reachable(i, i));
    }
}

#[test]
fn shortest_path_on_a_line() {
    let g = line_graph();
    assert_eq!(g.distance(0, 3), Some(3.0));
    assert_eq!(g.distance(1, 2), Some(1.0));
    assert_eq!(g.distance(0, 0), Some(0.0));
}

#[test]
fn unreachable_distance_is_none() {
    let g = line_graph();
    assert_eq!(g.distance(3, 0), None);
}

#[test]
fn shortest_path_prefers_cheaper_route() {
    // 0->1 direct costs 10; 0->2->1 costs 1+1=2.
    let mut g = TensorGraph::new(3);
    g.add_edge(0, 1, 10.0);
    g.add_edge(0, 2, 1.0);
    g.add_edge(2, 1, 1.0);
    assert_eq!(g.distance(0, 1), Some(2.0));
}

#[test]
fn walk_counts_count_paths() {
    // Diamond: 0->1, 0->2, 1->3, 2->3. Two 2-edge walks from 0 to 3.
    let mut g = TensorGraph::new(4);
    g.add_edge_unweighted(0, 1);
    g.add_edge_unweighted(0, 2);
    g.add_edge_unweighted(1, 3);
    g.add_edge_unweighted(2, 3);
    let w2 = g.walk_counts(2);
    assert_eq!(*w2.get(0, 3).unwrap(), 2.0);
    assert_eq!(*w2.get(0, 0).unwrap(), 0.0);
}

#[test]
fn walk_counts_zero_is_identity() {
    let g = line_graph();
    let w0 = g.walk_counts(0);
    assert_eq!(*w0.get(2, 2).unwrap(), 1.0);
    assert_eq!(*w0.get(0, 1).unwrap(), 0.0);
}

#[test]
fn cycle_is_mutually_reachable() {
    let mut g = TensorGraph::new(3);
    g.add_edge_unweighted(0, 1);
    g.add_edge_unweighted(1, 2);
    g.add_edge_unweighted(2, 0);
    for u in 0..3 {
        for v in 0..3 {
            assert!(g.reachable(u, v), "{u}->{v}");
        }
    }
}

#[test]
fn boolean_matmul_is_or_and() {
    let a = PaddedTileLattice::from_dense(2, 2, &[1.0, 0.0, 0.0, 1.0], Geometry::TPU_V).unwrap();
    let b = PaddedTileLattice::from_dense(2, 2, &[1.0, 1.0, 1.0, 0.0], Geometry::TPU_V).unwrap();
    let c = semiring_matmul::<Boolean>(&a, &b).unwrap();
    assert_eq!(c.to_dense(), vec![1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn tropical_matmul_is_min_plus() {
    let inf = f32::INFINITY;
    let a = PaddedTileLattice::from_dense(1, 2, &[1.0, 4.0], Geometry::TPU_V).unwrap();
    let b = PaddedTileLattice::from_dense(2, 1, &[2.0, 1.0], Geometry::TPU_V).unwrap();
    // min(1+2, 4+1) = min(3, 5) = 3.
    let c = semiring_matmul::<Tropical>(&a, &b).unwrap();
    assert_eq!(c.to_dense(), vec![3.0]);
    let _ = inf;
}
