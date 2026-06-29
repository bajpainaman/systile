//! Tests for the systolic matmul simulator. Correctness is checked against a
//! naive triple loop so the tiled dataflow can never silently diverge.

use systile::error::LatticeError;
use systile::{Bf16, Geometry, PaddedTileLattice};

