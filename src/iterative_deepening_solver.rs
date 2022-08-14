use std::collections::HashMap;

use crate::util;
use crate::util::BOARD_SIZE;

use util::SquareFlags;
use util::BOARD_COLS;
use util::BOARD_ROWS;

enum Direction {
    NORTH = 0,
    SOUTH = 1,
    EAST = 2,
    WEST = 3,
}

impl Direction {
    const VALUES: [Direction; 4] = [Direction::NORTH, Direction::SOUTH, Direction::EAST, Direction::WEST];
}
pub struct IterativeDeepeningSolver {
    min_costs: [u8; util::BOARD_SIZE],
    target_square: usize,
    board: [SquareFlags; util::BOARD_SIZE],
    bots: [usize; util::BOT_COUNT],
}

/**
 * Associated constructors
 */
impl IterativeDeepeningSolver {
    pub fn new() -> IterativeDeepeningSolver {
        return IterativeDeepeningSolver {
            min_costs: [0; util::BOARD_SIZE],
            target_square: 0,
            board: [SquareFlags::empty(); util::BOARD_SIZE],
            bots: [0; util::BOT_COUNT],
        };
    }

    /**
     * The with_values constructor accepts a board, a list of bots, and a target_square.
     * It is assumed that the first bot in the list of bots is the one that needs to reach the target square
     */
    pub fn with_values(
        board: [SquareFlags; util::BOARD_SIZE],
        bots: [usize; util::BOT_COUNT],
        target_square: usize,
    ) -> IterativeDeepeningSolver {
        let mut solver = IterativeDeepeningSolver {
            min_costs: [u8::MAX; util::BOARD_SIZE],
            target_square,
            board: board,
            bots: bots,
        };
        for square in solver.board.iter_mut() {
            square.set(SquareFlags::OCCUPIED, false);
        }
        solver.compute_min_moves();
        solver.sort_bots();
        for bot in solver.bots {
            solver.board[bot].set(SquareFlags::OCCUPIED, true);
        }
        return solver;
    }
}

/**
 * Methods
 */
impl IterativeDeepeningSolver {
    fn find_boundary_square(&self, start_square: usize, direction: Direction) -> usize {
        let mut square = start_square;
        match direction {
            Direction::NORTH => {
                while square >= BOARD_COLS && !self.board[square].contains(SquareFlags::WALL_NORTH)
                {
                    square -= BOARD_COLS;
                    if self.board[square].contains(SquareFlags::OCCUPIED) {
                        square += BOARD_COLS;
                        break;
                    }
                }
            }
            Direction::SOUTH => {
                while square < BOARD_SIZE - BOARD_COLS
                    && !self.board[square].contains(SquareFlags::WALL_SOUTH)
                {
                    square += BOARD_COLS;
                    if self.board[square].contains(SquareFlags::OCCUPIED) {
                        square -= BOARD_COLS;
                        break;
                    }
                }
            }
            Direction::EAST => {
                let row_bound = ((start_square / BOARD_ROWS) * BOARD_COLS) + BOARD_COLS - 1;
                while square < row_bound && !self.board[square].contains(SquareFlags::WALL_EAST) {
                    square += 1;
                    if self.board[square].contains(SquareFlags::OCCUPIED) {
                        square -= 1;
                        break;
                    }
                }
            }
            Direction::WEST => {
                let row_bound = (start_square / BOARD_ROWS) * BOARD_COLS;
                while square > row_bound && !self.board[square].contains(SquareFlags::WALL_WEST) {
                    square -= 1;
                    if self.board[square].contains(SquareFlags::OCCUPIED) {
                        square += 1;
                        break;
                    }
                }
            }
        }
        return square;
    }

    fn compute_min_moves(&mut self) {
        use std::collections::VecDeque;
        let mut visit_queue: VecDeque<(usize, u8)> = VecDeque::new();
        let mut visited = [false; util::BOARD_SIZE];
        visit_queue.push_back((self.target_square, 0));
        let queue_visit = |i: usize, moves: u8, visit_queue: &mut VecDeque<(usize, u8)>, visited: &mut [bool; util::BOARD_SIZE]| {
            if !visited[i] {
                visit_queue.push_back((i, moves + 1));
                visited[i] = true;
            }
        };
        while let Some(node) = visit_queue.pop_front() {
            let (square, moves) = node;
            self.min_costs[square] = moves;
            visited[square] = true;
            let northern_bound = self.find_boundary_square(square, Direction::NORTH);
            let southern_bound = self.find_boundary_square(square, Direction::SOUTH);
            let eastern_bound = self.find_boundary_square(square, Direction::EAST);
            let western_bound = self.find_boundary_square(square, Direction::WEST);

            let mut i = square;
            while i > northern_bound {
                i -= util::BOARD_COLS;
                queue_visit(i, moves, &mut visit_queue, &mut visited);
            }

            i = square;
            while i < southern_bound {
                i += util::BOARD_COLS;
                queue_visit(i, moves, &mut visit_queue, &mut visited);
            }

            i = square;
            while i < eastern_bound {
                i += 1;
                queue_visit(i, moves, &mut visit_queue, &mut visited);
            }

            i = square;
            while i > western_bound {
                i -= 1;
                queue_visit(i, moves, &mut visit_queue, &mut visited);
            }

        }
    }

    fn sort_bots(&mut self) {
        // Sort all but the first bot, which is the target bot and is exempt from the ordering rules
        if self.bots[1] > self.bots[2] {
            self.bots.swap(1, 2);
        }
        if self.bots[2] > self.bots[3] {
            self.bots.swap(2,3);
        }
        if self.bots[1] > self.bots[2] {
            self.bots.swap(1, 2);
        }
    }

    fn move_bot(&mut self, index: usize, direction: Direction) -> (usize, usize) {
        let original = self.bots[index];
        let boundary_square = self.find_boundary_square(self.bots[index], direction);
        self.board[self.bots[index]].set(SquareFlags::OCCUPIED, false);
        self.board[boundary_square].set(SquareFlags::OCCUPIED, true);
        self.bots[index] = boundary_square;
        if index != 0 {
            self.sort_bots();
        }
        return (original, boundary_square);
    }

    fn create_hash(&self) -> u32 {
        let mut hash: u32 = 0;
        for i in 0..util::BOT_COUNT {
            hash |= (self.bots[i] as u32) << (i * 8);
        }
        return hash;
    }

    fn is_solved(&self) -> bool {
        return self.bots[0] == self.target_square;
    }

    fn undo_move(&mut self, bot_move: (usize, usize)) {
        let (original, to_revert) = bot_move;
        let bot_index = self.bots.iter().position(|&x |{x == to_revert}).unwrap();
        self.bots[bot_index] = original;
        self.board[to_revert].set(SquareFlags::OCCUPIED, false);
        self.board[original].set(SquareFlags::OCCUPIED, true);
        if bot_index != 0 {
            self.sort_bots();
        }
    }

    pub fn solve(&mut self) -> bool {
        let mut previous_states: HashMap<u32, u8> = HashMap::new();
        fn helper(depth: u8, max_depth: u8, solver: &mut IterativeDeepeningSolver,  prev_states: &mut HashMap<u32, u8>) -> bool {
            let clearance = max_depth - depth;
            prev_states.insert(solver.create_hash(), clearance);
            if solver.is_solved() {
                println!("Found solution at depth {:?}", depth);
                return true;
            }
            if depth == max_depth {
                return false;
            }

            for bot_index in 0..util::BOT_COUNT {
                for direction in Direction::VALUES {
                    let bot_move = solver.move_bot(bot_index, direction);
                    let hash = solver.create_hash();
                    if let Some(&x) = prev_states.get(&hash) {
                        if x >= clearance - 1 {
                            solver.undo_move(bot_move);
                            continue;
                        }
                    }
                    if depth + 1 + solver.min_costs[solver.bots[0]] > max_depth {
                        solver.undo_move(bot_move);
                        continue;
                    }
                    let res = helper(depth + 1, max_depth, solver, prev_states);
                    if res {
                        println!("{:?}", solver.bots);
                        solver.undo_move(bot_move);
                        return true
                    }
                    solver.undo_move(bot_move);
                }
            }
            return false
        }
        let mut max_depth = self.min_costs[self.bots[0]];
        let mut res = false;
        while !res {
            res = helper(0, max_depth, self, &mut previous_states);
            max_depth += 1;
        }
        println!("{:?}", self.bots);
        return res;
    }
}

#[cfg(test)]
mod solver_tests {
    use super::*;
    use crate::util::BOT_COUNT;

    #[test]
    fn solve() {
        let mut solver = IterativeDeepeningSolver::new();
        assert!(solver.solve());
        assert_eq!(solver.board.len(), 256);
    }

    #[test]
    fn find_boundary_square() {
        let solver = IterativeDeepeningSolver::new();
        // Top Left Corner
        assert_eq!(solver.find_boundary_square(0x00, Direction::NORTH), 0x00);
        assert_eq!(solver.find_boundary_square(0x00, Direction::SOUTH), 0xF0);
        assert_eq!(solver.find_boundary_square(0x00, Direction::WEST), 0x00);
        assert_eq!(solver.find_boundary_square(0x00, Direction::EAST), 0x0F);

        // Top Right Corner
        assert_eq!(solver.find_boundary_square(0x0F, Direction::NORTH), 0x0F);
        assert_eq!(solver.find_boundary_square(0x0F, Direction::SOUTH), 0xFF);
        assert_eq!(solver.find_boundary_square(0x0F, Direction::WEST), 0x00);
        assert_eq!(solver.find_boundary_square(0x0F, Direction::EAST), 0x0F);

        // Bottom Left Corner
        assert_eq!(solver.find_boundary_square(0xF0, Direction::NORTH), 0x00);
        assert_eq!(solver.find_boundary_square(0xF0, Direction::SOUTH), 0xF0);
        assert_eq!(solver.find_boundary_square(0xF0, Direction::WEST), 0xF0);
        assert_eq!(solver.find_boundary_square(0xF0, Direction::EAST), 0xFF);

        // Bottom Right Corner
        assert_eq!(solver.find_boundary_square(0xFF, Direction::NORTH), 0x0F);
        assert_eq!(solver.find_boundary_square(0xFF, Direction::SOUTH), 0xFF);
        assert_eq!(solver.find_boundary_square(0xFF, Direction::WEST), 0xF0);
        assert_eq!(solver.find_boundary_square(0xFF, Direction::EAST), 0xFF);

        // Central Square
        assert_eq!(solver.find_boundary_square(0xAA, Direction::NORTH), 0x0A);
        assert_eq!(solver.find_boundary_square(0xAA, Direction::SOUTH), 0xFA);
        assert_eq!(solver.find_boundary_square(0xAA, Direction::WEST), 0xA0);
        assert_eq!(solver.find_boundary_square(0xAA, Direction::EAST), 0xAF);
    }

    #[test]
    fn find_boundary_square_east_wall() {
        let mut squares = [SquareFlags::empty(); BOARD_SIZE];
        squares[0x05].toggle(SquareFlags::WALL_EAST);
        let solver = IterativeDeepeningSolver::with_values(squares, [0; BOT_COUNT], 0);
        assert_eq!(solver.find_boundary_square(0x00, Direction::EAST), 0x05);
    }

    #[test]
    fn find_boundary_square_west_wall() {
        let mut squares = [SquareFlags::empty(); BOARD_SIZE];
        squares[0x05].toggle(SquareFlags::WALL_WEST);
        let solver = IterativeDeepeningSolver::with_values(squares, [0; BOT_COUNT], 0);
        assert_eq!(solver.find_boundary_square(0x0F, Direction::WEST), 0x05);
    }

    #[test]
    fn find_boundary_square_north_wall() {
        let mut squares = [SquareFlags::empty(); BOARD_SIZE];
        squares[0x50].toggle(SquareFlags::WALL_NORTH);
        let solver = IterativeDeepeningSolver::with_values(squares, [0; BOT_COUNT], 0);
        assert_eq!(solver.find_boundary_square(0xF0, Direction::NORTH), 0x50);
    }

    #[test]
    fn find_boundary_square_south_wall() {
        let mut squares = [SquareFlags::empty(); BOARD_SIZE];
        squares[0x50].toggle(SquareFlags::WALL_SOUTH);
        let solver = IterativeDeepeningSolver::with_values(squares, [0; BOT_COUNT], 0);
        assert_eq!(solver.find_boundary_square(0x00, Direction::SOUTH), 0x50);
    }

    #[test]
    fn find_boundary_square_occupied() {
        let squares = [SquareFlags::empty(); BOARD_SIZE];
        let mut solver = IterativeDeepeningSolver::with_values(squares, [0; BOT_COUNT], 0);
        solver.board[0xAA].toggle(SquareFlags::OCCUPIED);

        assert_eq!(solver.find_boundary_square(0x0A, Direction::SOUTH), 0x9A);
        assert_eq!(solver.find_boundary_square(0xFA, Direction::NORTH), 0xBA);
        assert_eq!(solver.find_boundary_square(0xA0, Direction::EAST), 0xA9);
        assert_eq!(solver.find_boundary_square(0xAF, Direction::WEST), 0xAB);
    }

    #[test]
    fn sort_bots_sorts_all_but_last_bot() {
        let squares = [SquareFlags::empty(); BOARD_SIZE];
        let mut solver = IterativeDeepeningSolver::with_values(squares, [0; util::BOT_COUNT], 0);
        solver.bots = [3, 1, 2, 0];

        solver.sort_bots();
        assert_eq!(solver.bots, [3, 0, 1, 2]);
    }

    #[test]
    fn board_with_values_initializes_correctly() {
        let squares = [SquareFlags::empty(); BOARD_SIZE];
        let solver = IterativeDeepeningSolver::with_values(squares, [3, 2, 1, 0], 0);

        assert_eq!(solver.bots, [3, 0, 1, 2]);
        for (i, &cost) in solver.min_costs.iter().enumerate() {
            if i == 0 {
                assert_eq!(cost, 0);
            } else if i < util::BOARD_COLS || i % util::BOARD_COLS == 0 {
                assert_eq!(cost, 1);
            } else {
                assert_eq!(cost, 2);
            }
        }
        for bot in solver.bots {
            assert!(solver.board[bot].contains(SquareFlags::OCCUPIED));
        }
    }

    #[test]
    fn move_bot_works_as_expected() {
        let mut solver = IterativeDeepeningSolver::with_values([SquareFlags::empty(); util::BOARD_SIZE], [3, 2, 1, 0], 0);
        assert_eq!(solver.bots, [3, 0, 1, 2]);
        assert!(solver.board[0].contains(SquareFlags::OCCUPIED));
        solver.move_bot(1, Direction::SOUTH);
        assert_eq!(solver.bots, [3, 1, 2, 240]);
        assert!(!solver.board[0].contains(SquareFlags::OCCUPIED));
        assert!(solver.board[240].contains(SquareFlags::OCCUPIED));
    }

    #[test]
    fn undo_move_works_as_expected() {
        let mut solver = IterativeDeepeningSolver::with_values([SquareFlags::empty(); util::BOARD_SIZE], [3, 2, 1, 0], 0);
        assert_eq!(solver.bots, [3, 0, 1, 2]);
        assert!(solver.board[0].contains(SquareFlags::OCCUPIED));
        let bot_move = solver.move_bot(1, Direction::SOUTH);
        assert_eq!(solver.bots, [3, 1, 2, 240]);
        assert!(!solver.board[0].contains(SquareFlags::OCCUPIED));
        assert!(solver.board[240].contains(SquareFlags::OCCUPIED));
        solver.undo_move(bot_move);
        assert_eq!(solver.bots, [3, 0, 1, 2]);
        assert!(!solver.board[240].contains(SquareFlags::OCCUPIED));
        assert!(solver.board[0].contains(SquareFlags::OCCUPIED));
    }

    #[test]
    fn solves_correctly() {
        // let raw_board: [u8; util::BOARD_SIZE] = [
        //     09, 01, 05, 01, 03, 09, 01, 01, 01, 03, 09, 01, 01, 01, 01, 03, 
        //     08, 02, 09, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 06, 08, 06, 
        //     08, 00, 00, 00, 00, 00, 00, 00, 00, 04, 00, 00, 00, 01, 00, 03,
        //     08, 00, 00, 00, 00, 02, 12, 00, 02, 09, 00, 00, 00, 00, 04, 02,
        //     12, 00, 00, 00, 04, 00, 01, 00, 00, 00, 00, 00, 00, 00, 03, 10,
        //     09, 00, 00, 00, 03, 08, 00, 00, 00, 00, 00, 00, 00, 00, 00, 02,
        //     08, 06, 08, 00, 00, 00, 00, 04, 04, 00, 00, 02, 12, 00, 00, 02, 
        //     08, 01, 00, 00, 00, 00, 02, 09, 03, 08, 00, 00, 01, 00, 00, 02, 
        //     08, 00, 04, 00, 02, 12, 02, 12, 06, 08, 00, 00, 00, 00, 00, 06, 
        //     08, 18, 09, 00, 00, 01, 00, 01, 01, 00, 00, 00, 00, 04, 00, 03, 
        //     08, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 02, 09, 00, 02, 
        //     28, 00, 00, 00, 00, 00, 00, 00, 00, 00, 06, 08, 00, 00, 00, 02, 
        //     09, 00, 00, 00, 04, 00, 00, 00, 00, 00, 01, 00, 00, 02, 12, 02, 
        //     08, 00, 00, 16, 03, 08, 00, 00, 00, 04, 00, 00, 00, 00, 01, 02,
        //     08, 06, 08, 00, 00, 00, 00, 00, 00, 03, 08, 00, 00, 00, 16, 02, 
        //     12, 05, 04, 04, 04, 06, 12, 04, 04, 04, 04, 06, 12, 04, 04, 06,
        // ];
        let raw_board = [9, 1, 1, 1, 3, 9, 1, 1, 1, 3, 9, 1, 1, 1, 5, 3, 8, 0, 22, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 9, 2, 8, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 28, 0, 0, 0, 2, 26, 12, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 6, 12, 1, 0, 0, 0, 2, 9, 0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8, 0, 0, 0, 0, 3, 8, 4, 4, 0, 4, 0, 0, 6, 8, 2, 8, 0, 0, 0, 0, 0, 2, 9, 3, 8, 3, 8, 0, 1, 0, 2, 8, 0, 0, 4, 0, 0, 2, 12, 6, 8, 0, 0, 0, 0, 4, 2, 8, 0, 0, 3, 8, 0, 0, 1, 1, 0, 0, 0, 0, 2, 9, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 12, 0, 0, 0, 6, 10, 12, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 0, 0, 0, 3, 8, 1, 0, 0, 0, 0, 6, 8, 0, 3, 8, 0, 0, 0, 0, 2, 12, 0, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 9, 2, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 8, 2, 12, 4, 4, 4, 4, 6, 12, 4, 4, 4, 6, 12, 4, 5, 4, 6];
        let board = raw_board.map(|x|{ return SquareFlags::from_bits(x).unwrap() });
        let mut solver = IterativeDeepeningSolver::with_values(board,[43, 226, 48, 18], 201);
        solver.solve();
    }
}
