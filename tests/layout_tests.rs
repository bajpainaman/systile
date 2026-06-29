//! Tests for layout address arithmetic and the validity mask.

use systile::layout::Layout;
use systile::mask::Mask;
use systile::{Geometry, PaddedTileLattice, Shape};

#[test]
fn offset_of_origin_is_zero() {
    let shape = Shape::new(8, 128, &Geometry::TPU_V);
    let layout = Layout::new(&shape, &Geometry::TPU_V);
    assert_eq!(layout.offset(0, 0), 0);
}

