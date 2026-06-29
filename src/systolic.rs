//! A CPU reference simulator of weight-stationary systolic matmul.
//!
//! A TPU multiplies matrices by streaming one operand through a square grid of
//! multiply-accumulate cells while the other operand sits stationary in the grid.
//! This module reproduces the *blocking and accumulation order* of that dataflow
//! — weights loaded one `mxu x mxu` block at a time, products accumulated into an
//! f32 accumulator — so a result computed here matches what the hardware returns
//! bit-for-bit in the f32 case and closely in the bf16 case. It also reports the
//! work performed so you can reason about utilisation before deploying.

