//! Sort a vector with matrix multiplies: ranks are `C·1`, the sort is `P·x`.
//!
//! Run with `cargo run --release --example sort_by_matmul`.

use systile::prelude::*;

fn main() {
    let sorter = TensorSort::new();
    let x = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];

    println!("input:        {x:?}");
    println!("ranks (C·1):  {:?}", sorter.ranks(&x));
    println!("argsort:      {:?}", sorter.argsort(&x));

    let sorted = sorter.sort(&x);
    println!("sorted:       {sorted:?}");

    // The same result by literally multiplying the permutation matrix by x.
    let via_matmul = sorter.sort_via_matmul(&x);
    println!("P·x:          {via_matmul:?}");

    let mut expected = x.to_vec();
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(sorted, expected);
    assert_eq!(via_matmul, expected);
    println!("\n✓ sorted by comparison matmul — O(n²) dense work instead of branches.");
}
