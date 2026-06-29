//! Tile-level sparsity.
//!
//! Systolic hardware pays the same cost for a tile of zeros as for a tile of real
//! work, so the highest-value sparsity optimisation is to *skip whole tiles* that
//! are entirely zero. This module finds those tiles.

use crate::bf16::Bf16;
use crate::lattice::PaddedTileLattice;

