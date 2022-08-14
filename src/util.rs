use bitflags::bitflags;

pub const BOARD_ROWS: usize = 16;
pub const BOARD_COLS: usize = 16;
pub const BOARD_SIZE: usize = BOARD_ROWS * BOARD_COLS;
pub const BOT_COUNT: usize = 4;

bitflags! {
    pub struct SquareFlags: u8 {
        const WALL_NORTH = 0b00000001;
        const WALL_EAST  = 0b00000010;
        const WALL_SOUTH = 0b00000100;
        const WALL_WEST  = 0b00001000;
        const OCCUPIED   = 0b00010000;
    }
}

#[cfg(test)]
mod bitflags_test {
    use crate::util::SquareFlags;

    #[test]
    fn test_bitflags() {
        assert_eq!(core::mem::size_of::<SquareFlags>(), 1);
    }
}