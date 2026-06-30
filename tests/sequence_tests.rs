//! Tests for the holographic sequence.

use systile::HoloSequence;

#[test]
fn empty_sequence() {
    let s = HoloSequence::new(2048, 32, 1);
    assert!(s.is_empty());
    assert_eq!(s.len(), 0);
}

#[test]
fn push_increases_len() {
    let mut s = HoloSequence::new(2048, 32, 1);
    s.push(3);
    s.push(5);
    assert_eq!(s.len(), 2);
}

#[test]
fn single_element_reads_back() {
    let mut s = HoloSequence::new(4096, 64, 1);
    s.push(9);
    assert_eq!(s.get(0).0, 9);
}

#[test]
fn order_is_preserved() {
    let mut s = HoloSequence::new(8192, 64, 2);
    let seq = [5usize, 1, 9, 9, 2, 7];
    for &x in &seq {
        s.push(x);
    }
    for (pos, &x) in seq.iter().enumerate() {
        assert_eq!(s.get(pos).0, x, "position {pos}");
    }
}

#[test]
fn repeated_symbol_at_different_positions() {
    // The same symbol at two positions must both decode correctly — this is what
    // the permutation binding buys over a plain bundle.
    let mut s = HoloSequence::new(8192, 32, 3);
    s.push(4);
    s.push(4);
    s.push(4);
    assert_eq!(s.decode(), vec![4, 4, 4]);
}

#[test]
fn decode_matches_get() {
    let mut s = HoloSequence::new(8192, 64, 4);
    let seq = [1usize, 2, 3, 4, 5];
    for &x in &seq {
        s.push(x);
    }
    let decoded = s.decode();
    for (pos, id) in decoded.iter().enumerate() {
        assert_eq!(*id, s.get(pos).0);
    }
}

#[test]
fn decode_recovers_full_sequence() {
    let mut s = HoloSequence::new(8192, 100, 5);
    let seq = [10usize, 20, 30, 40, 50, 60, 70, 80];
    for &x in &seq {
        s.push(x);
    }
    assert_eq!(s.decode(), seq.to_vec());
}
