//! Tests for scaled dot-product attention.

use systile::TensorAttention;

fn identity_keys() -> Vec<Vec<f32>> {
    vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ]
}

#[test]
fn sharp_query_retrieves_its_value() {
    let attn = TensorAttention::new(4);
    let keys = identity_keys();
    let values = [vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let kr: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let vr: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
    // A large-magnitude query onto key 2 sharpens the softmax onto value 3.0.
    let out = attn.attend_one(&[0.0, 0.0, 12.0, 0.0], &kr, &vr);
    assert!((out[0] - 3.0).abs() < 0.5, "got {}", out[0]);
}

#[test]
fn uniform_query_averages_values() {
    let attn = TensorAttention::new(4);
    let keys = identity_keys();
    let values = [vec![10.0], vec![20.0], vec![30.0], vec![40.0]];
    let kr: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let vr: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
    // A zero query gives equal weights -> the mean of the values.
    let out = attn.attend_one(&[0.0, 0.0, 0.0, 0.0], &kr, &vr);
    assert!((out[0] - 25.0).abs() < 1e-3, "got {}", out[0]);
}

#[test]
fn output_dimension_is_value_dimension() {
    let attn = TensorAttention::new(4);
    let keys = identity_keys();
    let values = [
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
        vec![7.0, 8.0],
    ];
    let kr: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let vr: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
    let out = attn.attend_one(&[1.0, 0.0, 0.0, 0.0], &kr, &vr);
    assert_eq!(out.len(), 2);
}

#[test]
fn batch_of_queries() {
    let attn = TensorAttention::new(4);
    let keys = identity_keys();
    let values = [vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let kr: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let vr: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
    let q0 = [10.0f32, 0.0, 0.0, 0.0];
    let q1 = [0.0f32, 0.0, 0.0, 10.0];
    let queries: Vec<&[f32]> = vec![&q0, &q1];
    let out = attn.attend(&queries, &kr, &vr);
    assert_eq!(out.len(), 2);
    assert!((out[0][0] - 1.0).abs() < 0.5); // near value 0
    assert!((out[1][0] - 4.0).abs() < 0.5); // near value 3
}

#[test]
fn weights_form_a_convex_blend() {
    let attn = TensorAttention::new(4);
    let keys = identity_keys();
    let values = [vec![5.0], vec![5.0], vec![5.0], vec![5.0]];
    let kr: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let vr: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
    // All values equal -> output equals that value regardless of the query.
    let out = attn.attend_one(&[3.0, 1.0, 4.0, 1.0], &kr, &vr);
    assert!((out[0] - 5.0).abs() < 1e-4);
}

#[test]
fn empty_queries() {
    let attn = TensorAttention::new(4);
    let keys = identity_keys();
    let values = [vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let kr: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let vr: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
    assert!(attn.attend(&[], &kr, &vr).is_empty());
}
