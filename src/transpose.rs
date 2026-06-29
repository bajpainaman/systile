//! Transpose and relayout.
//!
//! On real hardware a transpose is a sequence of lane rotations through the
//! cross-lane unit; here we produce the logically-transposed lattice directly.
//! Relayout re-tiles the same logical data under a different [`Geometry`], which
//! is what you do when handing a tensor from the vector unit to the matrix unit.

