//! End-to-end holographic key→value store: build a phonebook entirely in
//! superposition, then recover every number with a single batched matmul.
//!
//! Run with `cargo run --release --example holo_kv`.

use systile::prelude::*;

fn main() {
    // 8192-dimensional space, value vocabulary of 1000 possible "numbers".
    let dim = 8192;
    let n_numbers = 1000;
    let mut book = HoloMemory::new(dim, n_numbers, 0xC0FFEE);

    // Insert 200 name->number bindings. Names are key ids 0..200; each maps to a
    // value id we choose. The whole map ends up summed into ONE 8192-vector.
    let entries: Vec<(usize, usize)> = (0..200).map(|i| (i, (i * 7 + 3) % n_numbers)).collect();
    for &(name, number) in &entries {
        book.insert(name, number);
    }

    println!("{book:?}");
    println!(
        "stored {} entries inside a single {}-dim vector ({} bytes)",
        book.len(),
        dim,
        dim * 4
    );
    println!(
        "estimated capacity: {:.0} entries",
        book.estimated_capacity()
    );

    // Look up ALL names at once — this is one (200 x 8192) . (8192 x 1000) matmul
    // through the systolic engine.
    let names: Vec<usize> = entries.iter().map(|&(n, _)| n).collect();
    let recovered = book.batch_get(&names);

    let correct = entries
        .iter()
        .zip(&recovered)
        .filter(|(&(_, want), &(got, _))| got == want)
        .count();

    println!(
        "\nrecovered {correct}/{} numbers correctly ({:.1}% recall) in one batched matmul",
        entries.len(),
        100.0 * correct as f64 / entries.len() as f64
    );
}
