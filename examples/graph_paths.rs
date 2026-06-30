//! Shortest paths and reachability as matrix powers over a semiring.
//!
//! Run with `cargo run --release --example graph_paths`.

use systile::prelude::*;

fn main() {
    // A small weighted DAG-ish graph.
    //   0 →1(4) →3(1)
    //   0 →2(1) →3(5)
    //   2 →1(2)
    let mut g = TensorGraph::new(4);
    g.add_edge(0, 1, 4.0);
    g.add_edge(0, 2, 1.0);
    g.add_edge(2, 1, 2.0);
    g.add_edge(1, 3, 1.0);
    g.add_edge(2, 3, 5.0);

    println!("{g:?}\n");

    println!("all-pairs shortest paths (tropical matrix powers):");
    for u in 0..g.nodes() {
        for v in 0..g.nodes() {
            match g.distance(u, v) {
                Some(d) => print!("  {u}->{v}:{d:>3}"),
                None => print!("  {u}->{v}:  ∞"),
            }
        }
        println!();
    }

    // Best route 0 -> 3 is 0->2->1->3 = 1 + 2 + 1 = 4, beating 0->1->3 = 5.
    println!(
        "\nshortest 0 -> 3 = {:?}  (via 0->2->1->3)",
        g.distance(0, 3)
    );
    assert_eq!(g.distance(0, 3), Some(4.0));

    println!("reachable 0 -> 3? {}", g.reachable(0, 3));
    println!("reachable 3 -> 0? {}", g.reachable(3, 0));
    println!("\n✓ traversal computed as a handful of dense matmuls, no queue, no branches.");
}
