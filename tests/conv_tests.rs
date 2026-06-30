//! Tests for the im2col convolution / pattern search.

use systile::TensorConv;

#[test]
fn correlation_length_is_valid() {
    let c = TensorConv::new();
    let scores = c.correlate(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 1.0]);
    assert_eq!(scores.len(), 4); // 5 - 2 + 1
}

#[test]
fn correlation_values_are_window_dot_kernel() {
    let c = TensorConv::new();
    // kernel [1,1] gives adjacent sums.
    let scores = c.correlate(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0]);
    assert_eq!(scores, vec![3.0, 5.0, 7.0]);
}

#[test]
fn best_offset_finds_peak() {
    let c = TensorConv::new();
    let pattern = [1.0f32, 2.0, 3.0];
    let signal = [0.0, 0.0, 1.0, 2.0, 3.0, 0.0];
    assert_eq!(c.best_offset(&signal, &pattern), Some(2));
}

#[test]
fn find_all_locates_every_occurrence() {
    let c = TensorConv::new();
    let pattern = [1.0f32, -1.0, 2.0];
    let signal = [0.0, 1.0, -1.0, 2.0, 0.0, 1.0, -1.0, 2.0];
    assert_eq!(c.find_all(&signal, &pattern), vec![1, 5]);
}

#[test]
fn find_all_rejects_near_misses() {
    let c = TensorConv::new();
    let pattern = [1.0f32, 2.0, 3.0];
    // A window with the same dot product but different values must not match.
    let signal = [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0];
    assert_eq!(c.find_all(&signal, &pattern), vec![4]);
}

#[test]
fn pattern_longer_than_signal_is_empty() {
    let c = TensorConv::new();
    assert!(c.correlate(&[1.0, 2.0], &[1.0, 2.0, 3.0]).is_empty());
    assert_eq!(c.best_offset(&[1.0], &[1.0, 2.0]), None);
}

#[test]
fn no_occurrence_returns_empty() {
    let c = TensorConv::new();
    assert!(c.find_all(&[0.0, 0.0, 0.0], &[1.0, 1.0]).is_empty());
}
