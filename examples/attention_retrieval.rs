//! Scaled dot-product attention as a soft associative memory.
//!
//! Run with `cargo run --release --example attention_retrieval`.

use systile::prelude::*;

fn main() {
    let attn = TensorAttention::new(4);

    // Four key/value pairs: each key is a one-hot-ish direction, each value a tag.
    let keys: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    let values: Vec<Vec<f32>> = vec![
        vec![10.0, 0.0],
        vec![20.0, 0.0],
        vec![30.0, 0.0],
        vec![40.0, 0.0],
    ];
    let kref: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let vref: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    // A query close to key 2 (scaled up so attention sharpens onto it).
    let query = [0.0f32, 0.0, 8.0, 0.0];
    let out = attn.attend_one(&query, &kref, &vref);
    println!("sharp query near key 2 -> {out:?}  (≈ value 2 = [30, 0])");
    assert!((out[0] - 30.0).abs() < 2.0);

    // A diffuse query leaning toward keys 0 and 1 -> a blend of all values, pulled
    // below the uniform mean (25) because keys 0 and 1 carry the most weight.
    let blend = attn.attend_one(&[1.0, 1.0, 0.0, 0.0], &kref, &vref);
    println!("diffuse query leaning to keys 0,1 -> {blend:?}  (blend pulled below 25)");
    assert!(blend[0] > 10.0 && blend[0] < 25.0);

    println!("\n✓ retrieval as softmax(QKᵀ/√d)·V — three matmuls, a soft k-NN.");
}
