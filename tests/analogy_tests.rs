//! Analogical reasoning via bind/unbind: the "Dollar of Mexico" construction.
//!
//! This is a regression test for the full VSA pipeline — records as bundles of
//! role⊛filler bindings, a mapping as the bind of two records, and a query
//! resolved by codebook cleanup.

use systile::{Codebook, Hyper};

const NAME: usize = 0;
const CAP: usize = 1;
const CUR: usize = 2;
const USA: usize = 3;
const WDC: usize = 4;
const DOLLAR: usize = 5;
const MEXICO: usize = 6;
const MEXICO_CITY: usize = 7;
const PESO: usize = 8;

fn record(book: &Codebook, dim: usize, slots: &[(usize, usize)]) -> Hyper {
    let mut acc = Hyper::zeros(dim);
    for &(role, filler) in slots {
        acc.bundle_into(&book.atom(role).bind(&book.atom(filler)));
    }
    acc
}

#[test]
fn dollar_of_mexico_is_peso() {
    let dim = 10000;
    let book = Codebook::new(dim, 9, 0x1CE);
    let usa = record(&book, dim, &[(NAME, USA), (CAP, WDC), (CUR, DOLLAR)]);
    let mexico = record(
        &book,
        dim,
        &[(NAME, MEXICO), (CAP, MEXICO_CITY), (CUR, PESO)],
    );

    let mapping = usa.bind(&mexico);
    let query = mapping.bind(&book.atom(DOLLAR));
    assert_eq!(book.cleanup(&query).0, PESO);
}

#[test]
fn capital_of_mexico_is_mexico_city() {
    let dim = 10000;
    let book = Codebook::new(dim, 9, 0x1CE);
    let usa = record(&book, dim, &[(NAME, USA), (CAP, WDC), (CUR, DOLLAR)]);
    let mexico = record(
        &book,
        dim,
        &[(NAME, MEXICO), (CAP, MEXICO_CITY), (CUR, PESO)],
    );

    let mapping = usa.bind(&mexico);
    // Map Washington through the analogy -> Mexico City.
    let query = mapping.bind(&book.atom(WDC));
    assert_eq!(book.cleanup(&query).0, MEXICO_CITY);
}

#[test]
fn single_record_field_lookup() {
    let dim = 8192;
    let book = Codebook::new(dim, 9, 0x1CE);
    let usa = record(&book, dim, &[(NAME, USA), (CAP, WDC), (CUR, DOLLAR)]);
    // Unbinding a role returns its filler.
    assert_eq!(book.cleanup(&usa.bind(&book.atom(CUR))).0, DOLLAR);
}
