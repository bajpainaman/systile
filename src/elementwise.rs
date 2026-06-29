//! Element-wise maps and binary combinators over the logical region.
//!
//! These operate only on logical elements; padding is regenerated as
//! `T::default()` so it can never leak into a result. On hardware these map to
//! the vector unit, which is why they preserve tiling exactly.

