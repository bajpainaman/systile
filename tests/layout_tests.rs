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

#[test]
fn within_tile_is_row_major_sublane_lane() {
    let shape = Shape::new(8, 128, &Geometry::TPU_V);
    let layout = Layout::new(&shape, &Geometry::TPU_V);
    // Same tile: moving one column moves one slot; moving one sublane moves `lanes`.
    assert_eq!(layout.offset(0, 1), 1);
    assert_eq!(layout.offset(1, 0), 128);
}

#[test]
fn second_tile_starts_after_first() {
    let shape = Shape::new(8, 256, &Geometry::TPU_V);
    let layout = Layout::new(&shape, &Geometry::TPU_V);
    assert_eq!(layout.offset(0, 128), Geometry::TPU_V.tile_len());
}

#[test]
fn tile_base_is_multiple_of_tile_len() {
    let shape = Shape::new(16, 256, &Geometry::TPU_V);
    let layout = Layout::new(&shape, &Geometry::TPU_V);
    assert_eq!(layout.tile_base(1, 1), 3 * Geometry::TPU_V.tile_len());
}

#[test]
fn mask_marks_logical_region() {
    let shape = Shape::new(3, 5, &Geometry::TPU_V);
    let mask = Mask::from_shape(&shape);
    assert!(mask.is_valid(2, 4));
    assert!(!mask.is_valid(3, 0));
    assert!(!mask.is_valid(0, 5));
}

#[test]
fn mask_counts_match_shape() {
    let shape = Shape::new(3, 5, &Geometry::TPU_V);
    let mask = Mask::from_shape(&shape);
    assert_eq!(mask.count_valid(), 15);
    assert_eq!(mask.count_padding(), 8 * 128 - 15);
}

#[test]
fn exact_shape_has_full_mask() {
    let shape = Shape::new(8, 128, &Geometry::TPU_V);
    let mask = Mask::from_shape(&shape);
    assert!(mask.is_full());
}

#[test]
fn mask_tile_count_matches_manual() {
    let shape = Shape::new(3, 5, &Geometry::TPU_V);
    let mask = Mask::from_shape(&shape);
    // The single tile holds the whole 3x5 logical block: 15 valid slots.
    assert_eq!(mask.count_valid_in_tile(0, 0, 8, 128), 15);
}

#[test]
fn iter_tiles_yields_num_tiles() {
    let l = PaddedTileLattice::<f32>::zeroed(16, 256, Geometry::TPU_V).unwrap();
    assert_eq!(l.iter_tiles().count(), l.num_tiles());
}

