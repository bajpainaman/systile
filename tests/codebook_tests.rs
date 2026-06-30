//! Tests for the matmul-backed codebook cleanup.

use systile::{Codebook, Hyper};

#[test]
fn codebook_reports_size() {
    let cb = Codebook::new(1024, 64, 1);
    assert_eq!(cb.dim(), 1024);
    assert_eq!(cb.len(), 64);
    assert!(!cb.is_empty());
}

#[test]
fn atom_matches_hyper_atom() {
    let cb = Codebook::new(512, 16, 99);
    // The codebook's atoms are regenerable and stable.
    assert_eq!(cb.atom(3), cb.atom(3));
    assert_ne!(cb.atom(3), cb.atom(4));
}

#[test]
fn clean_atom_cleans_up_to_itself() {
    let cb = Codebook::new(2048, 128, 7);
    for id in [0usize, 5, 42, 127] {
        let (best, _) = cb.cleanup(&cb.atom(id));
        assert_eq!(best, id, "atom {id} cleaned up to {best}");
    }
}

#[test]
fn cleanup_score_of_clean_atom_is_dim() {
    let cb = Codebook::new(1024, 32, 3);
    let (_, score) = cb.cleanup(&cb.atom(10));
    assert!((score - 1024.0).abs() < 1.0, "score was {score}");
}

#[test]
fn noisy_atom_still_cleans_up() {
    let cb = Codebook::new(8192, 64, 11);
    let mut noisy = cb.atom(20);
    // Flip ~10% of the coordinates.
    {
        let mut flipped = noisy.into_vec();
        for (i, x) in flipped.iter_mut().enumerate() {
            if i % 10 == 0 {
                *x = -*x;
            }
        }
        noisy = Hyper::from_vec(flipped);
    }
    assert_eq!(cb.cleanup(&noisy).0, 20);
}

#[test]
fn batch_cleanup_matches_individual() {
    let cb = Codebook::new(2048, 100, 5);
    let queries: Vec<Hyper> = (0..50).map(|i| cb.atom(i)).collect();
    let batch = cb.cleanup_batch(&queries);
    for (i, (best, _)) in batch.iter().enumerate() {
        assert_eq!(*best, i);
    }
}

#[test]
fn batch_cleanup_equals_single_cleanup() {
    let cb = Codebook::new(1024, 40, 8);
    let queries: Vec<Hyper> = (0..40).map(|i| cb.atom(i)).collect();
    let batch = cb.cleanup_batch(&queries);
    for (i, q) in queries.iter().enumerate() {
        assert_eq!(cb.cleanup(q).0, batch[i].0);
    }
}

#[test]
fn empty_batch_returns_empty() {
    let cb = Codebook::new(256, 8, 1);
    assert!(cb.cleanup_batch(&[]).is_empty());
}

#[test]
fn scores_have_count_columns() {
    let cb = Codebook::new(512, 17, 2);
    let scores = cb.scores_batch(&[cb.atom(0), cb.atom(1)]);
    assert_eq!(scores.len(), 2);
    assert_eq!(scores[0].len(), 17);
}

#[test]
fn matrix_is_dim_by_count() {
    let cb = Codebook::new(512, 20, 1);
    assert_eq!(cb.matrix().rows(), 512);
    assert_eq!(cb.matrix().cols(), 20);
}

#[test]
fn bf16_cleanup_recovers_clean_atoms() {
    let cb = Codebook::new(2048, 64, 21);
    let queries: Vec<Hyper> = (0..64).map(|i| cb.atom(i)).collect();
    let hits = cb.cleanup_batch_bf16(&queries);
    for (i, (best, _)) in hits.iter().enumerate() {
        assert_eq!(*best, i, "bf16 cleanup missed atom {i}");
    }
}

#[test]
fn bf16_cleanup_matches_f32_on_clean_atoms() {
    let cb = Codebook::new(1024, 32, 22);
    let queries: Vec<Hyper> = (0..32).map(|i| cb.atom(i)).collect();
    let f32_hits = cb.cleanup_batch(&queries);
    let bf16_hits = cb.cleanup_batch_bf16(&queries);
    for (a, b) in f32_hits.iter().zip(&bf16_hits) {
        assert_eq!(a.0, b.0);
    }
}

#[test]
fn bf16_empty_batch_is_empty() {
    let cb = Codebook::new(256, 8, 1);
    assert!(cb.cleanup_batch_bf16(&[]).is_empty());
}
