//! PageRank by power iteration — each step is one matmul.
//!
//! Run with `cargo run --release --example pagerank_demo`.

use systile::prelude::*;

fn main() {
    // A small link graph: node 0 is pointed to by everyone, node 4 dangles.
    //   1→0, 2→0, 3→0, 2→1, 3→1, 3→2, 0→2, 4 has no out-links.
    let mut pr = TensorPageRank::new(5);
    pr.add_link(1, 0);
    pr.add_link(2, 0);
    pr.add_link(3, 0);
    pr.add_link(2, 1);
    pr.add_link(3, 1);
    pr.add_link(3, 2);
    pr.add_link(0, 2);

    println!("{pr:?}");
    let ranks = pr.rank_default();

    let mut order: Vec<usize> = (0..ranks.len()).collect();
    order.sort_by(|&a, &b| ranks[b].partial_cmp(&ranks[a]).unwrap());

    println!("\nPageRank (descending):");
    for &node in &order {
        println!("  node {node}: {:.4}", ranks[node]);
    }

    let sum: f32 = ranks.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "ranks must sum to 1");
    // Node 0 receives the most links and should rank highest.
    assert_eq!(order[0], 0);
    println!("\n✓ stationary distribution found by repeated M·r matmuls.");
}
