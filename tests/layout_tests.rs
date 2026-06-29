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

#[test]
fn offset_and_coord_are_inverses() {
    let shape = Shape::new(16, 256, &Geometry::TPU_V);
    let layout = Layout::new(&shape, &Geometry::TPU_V);
    for row in 0..16 {
        for col in 0..256 {
            let off = layout.offset(row, col);
            assert_eq!(layout.coord(off), (row, col));
        }
    }
}

