//! "What is the Dollar of Mexico?" — Kanerva's classic demonstration that a
//! holographic representation can do analogical reasoning with *no training*, just
//! bind/unbind algebra and a single matmul cleanup.
//!
//! Two country records are each `(NAME⊛country) + (CAP⊛capital) + (CUR⊛currency)`.
//! Binding the two records together produces a mapping that translates one
//! country's slots into the other's — so unbinding "dollar" through it yields
//! "peso", purely by algebra.
//!
//! Run with `cargo run --example holo_analogy`.

use systile::prelude::*;

// Symbol ids in one shared codebook.
const NAME: usize = 0;
const CAP: usize = 1;
const CUR: usize = 2;
const USA: usize = 3;
const WDC: usize = 4;
const DOLLAR: usize = 5;
const MEXICO: usize = 6;
const MEXICO_CITY: usize = 7;
const PESO: usize = 8;

fn main() {
    let dim = 10000;
    let book = Codebook::new(dim, 9, 0x1CE);
    let atom = |id: usize| book.atom(id);

    // A record bundles role⊛filler bindings into one vector.
    let record = |slots: &[(usize, usize)]| {
        let mut acc = Hyper::zeros(dim);
        for &(role, filler) in slots {
            acc.bundle_into(&atom(role).bind(&atom(filler)));
        }
        acc
    };

    let usa = record(&[(NAME, USA), (CAP, WDC), (CUR, DOLLAR)]);
    let mexico = record(&[(NAME, MEXICO), (CAP, MEXICO_CITY), (CUR, PESO)]);

    // Bind the records: this maps USA's fillers to Mexico's, slot by slot.
    let mapping = usa.bind(&mexico);

    // "Dollar of Mexico" = unbind dollar through the mapping, then clean up.
    let query = mapping.bind(&atom(DOLLAR));
    let (answer, score) = book.cleanup(&query);

    let names = [
        "NAME",
        "CAP",
        "CUR",
        "usa",
        "washington",
        "dollar",
        "mexico",
        "mexico_city",
        "peso",
    ];
    println!("What is the Dollar of Mexico?");
    println!("  -> {} (score {:.0})", names[answer], score);
    assert_eq!(answer, PESO);
    println!("\n✓ analogical answer with zero training — bind, unbind, one matmul cleanup.");
}
