//! Show what bf16's seven mantissa bits do to a range of values.
//!
//! Run with `cargo run --example bf16_roundtrip`.

use systile::prelude::*;

fn main() {
    println!("{:>12}  {:>12}  {:>12}", "input", "bf16", "abs error");
    for &x in &[1.0f32, 0.1, 3.2, 1234.5, 1e-8, 65504.0, -2.5] {
        let b = Bf16::from_f32(x);
        let back = b.to_f32();
        println!("{x:>12}  {back:>12}  {:>12.3e}", (x - back).abs());
    }

    // bf16 keeps f32's exponent range, so very large and very small both survive.
    assert!(Bf16::from_f32(1e30).is_finite());
    assert!(Bf16::from_f32(1e-30).to_f32() > 0.0);
    println!("\nbf16 preserves f32 dynamic range, trades away mantissa precision.");
}
