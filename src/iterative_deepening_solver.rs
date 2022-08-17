use crate::util;

use std::collections::HashMap;
use ahash::{RandomState};

use util::SquareFlags;
use util::BOARD_COLS;

#[derive(Copy, Clone, PartialEq)]
enum Direction {
    North = 0,
    South = 1,
    East = 2,
    West = 3,
}

impl Direction {
    const VALUES: [Direction; 4] = [Direction::North, Direction::South, Direction::East, Direction::West];

    fn get_offset(&self) -> i32 {
        match self {
            Direction::North => -(BOARD_COLS as i32),
            Direction::South => BOARD_COLS as i32,
            Direction::East => 1,
            Direction::West => -1,
        }
    }

    fn get_wall(&self) -> SquareFlags {
        match self {
            Direction::North => SquareFlags::WALL_NORTH,
            Direction::South => SquareFlags::WALL_SOUTH,
            Direction::East => SquareFlags::WALL_EAST,
            Direction::West => SquareFlags::WALL_WEST,
        }
    }
}


pub struct IterativeDeepeningSolver {
    min_costs: [u8; util::BOARD_SIZE],
    target_square: u8,
    board: [SquareFlags; util::BOARD_SIZE],
    bots: [u8; util::BOT_COUNT],
    hash: u32,
}

/**
 * Associated constructors
 */
impl IterativeDeepeningSolver {
    /**
     * The with_values constructor accepts a board, a list of bots, and a target_square.
     * It is assumed that the first bot in the list of bots is the one that needs to reach the target square
     */
    pub fn with_values(
        board: [SquareFlags; util::BOARD_SIZE],
        bots: [u8; util::BOT_COUNT],
        target_square: u8,
    ) -> IterativeDeepeningSolver {
        let mut solver = IterativeDeepeningSolver {
            min_costs: [u8::MAX; util::BOARD_SIZE],
            hash: 0,
            target_square,
            board,
            bots,
        };
        for square in solver.board.iter_mut() {
            square.set(SquareFlags::OCCUPIED, false);
        }
        solver.compute_min_moves();
        solver.sort_bots();
        for bot in solver.bots {
            solver.board[bot as usize].set(SquareFlags::OCCUPIED, true);
        }
        solver
    }
}

/**
 * Methods
 */
impl IterativeDeepeningSolver {
    fn find_boundary_square(&self, start_square: u8, direction: Direction) -> u8 {
        let mut square = start_square as i32;
        let offset = direction.get_offset();
        let wall = direction.get_wall();
        while !self.board[square as usize].contains(wall) {
            square += offset;
            if self.board[square as usize].contains(SquareFlags::OCCUPIED) {
                square -= offset;
                break;
            }
        }
        square as u8
    }

    fn compute_min_moves(&mut self) {
        use std::collections::VecDeque;
        let mut visit_queue: VecDeque<(u8, u8)> = VecDeque::new();
        let mut visited = [false; util::BOARD_SIZE];
        visit_queue.push_back((self.target_square, 0));
        let queue_visit = |i: u8, moves: u8, visit_queue: &mut VecDeque<(u8, u8)>, visited: &mut [bool; util::BOARD_SIZE]| {
            if !visited[i as usize] {
                visit_queue.push_back((i, moves + 1));
                visited[i as usize] = true;
            }
        };
        while let Some(node) = visit_queue.pop_front() {
            let (square, moves) = node;
            self.min_costs[square as usize] = moves;
            visited[square as usize] = true;
            let northern_bound = self.find_boundary_square(square, Direction::North);
            let southern_bound = self.find_boundary_square(square, Direction::South);
            let eastern_bound = self.find_boundary_square(square, Direction::East);
            let western_bound = self.find_boundary_square(square, Direction::West);

            let mut i = square;
            while i > northern_bound {
                i -= util::BOARD_COLS as u8;
                queue_visit(i, moves, &mut visit_queue, &mut visited);
            }

            i = square;
            while i < southern_bound {
                i += util::BOARD_COLS as u8;
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
            self.bots.swap(2, 3);
        }
        if self.bots[1] > self.bots[2] {
            self.bots.swap(1, 2);
        }
        self.update_hash();
    }

    fn can_move (&self, bot: u8, direction: Direction) -> bool {
        let square = self.board[bot as usize];
        if square.contains(direction.get_wall()) {
            return false;
        }
        let next_square = (bot as i32 + direction.get_offset()) as usize;
        if self.board[next_square].contains(SquareFlags::OCCUPIED) {
            return false;
        }

        true
    }

    fn move_bot(&mut self, index: usize, direction: Direction) -> (u8, u8) {
        let original = self.bots[index];
        let boundary_square = self.find_boundary_square(self.bots[index], direction);
        self.board[self.bots[index] as usize ].set(SquareFlags::OCCUPIED, false);
        self.board[boundary_square as usize].set(SquareFlags::OCCUPIED, true);
        self.bots[index] = boundary_square as u8;
        self.sort_bots();
        (original, boundary_square)
    }

    fn update_hash(&mut self) {
        self.hash = 0;
        self.bots
            .iter()
            .enumerate()
            .for_each(|(i, bot)| self.hash |= (*bot as u32) << i * 8);
    }

    fn is_solved(&self) -> bool {
        self.bots[0] == self.target_square
    }

    fn undo_move(&mut self, bot_move: (u8, u8)) {
        let (original, to_revert) = bot_move;
        let bot_index = self.bots.iter().position(|&x| x == to_revert).unwrap();
        self.bots[bot_index] = original;
        self.board[to_revert as usize].set(SquareFlags::OCCUPIED, false);
        self.board[original as usize].set(SquareFlags::OCCUPIED, true);
        self.sort_bots();
    }

    pub fn solve(&mut self) -> Vec<(u8, u8)> {

        fn helper(depth: u8, max_depth: u8, solver: &mut IterativeDeepeningSolver,  prev_states: &mut HashMap<u32, u8, RandomState>, solution: &mut Vec<(u8, u8)>) -> bool {
            let clearance = max_depth - depth;
            prev_states.insert(solver.hash, clearance);
            if solver.is_solved() {
                return true;
            }

            for bot_index in 0..util::BOT_COUNT {
                for direction in Direction::VALUES {
                    if !solver.can_move(solver.bots[bot_index], direction) {
                        continue;
                    }
                    let bot_move = solver.move_bot(bot_index, direction);
                    if let Some(&x) = prev_states.get(&solver.hash) {
                        if x >= clearance - 1 {
                            solver.undo_move(bot_move);
                            continue;
                        }
                    }
                    if depth + 1 + solver.min_costs[solver.bots[0] as usize] > max_depth {
                        solver.undo_move(bot_move);
                        continue;
                    }
                    let res = helper(depth + 1, max_depth, solver, prev_states, solution);
                    solver.undo_move(bot_move);
                    if res {
                        solution.push(bot_move);
                        return true
                    }
                }
            }
            false
        }

        let mut max_depth = self.min_costs[self.bots[0] as usize];
        let mut previous_states: HashMap<u32, u8, RandomState> = HashMap::default();

        let mut solution: Vec<(u8, u8)> = Vec::new();

        while !helper(0, max_depth, self, &mut previous_states, &mut solution) {
            max_depth += 1;
        }
        solution.reverse();
        solution
    }
}

#[cfg(test)]
mod solver_tests {
    use super::*;
    #[test]
    fn solves_correctly() {
        let raw_board = [
            09, 01, 01, 01, 03, 09, 01, 01, 01, 03, 09, 01, 01, 01, 05, 03, 
            08, 00, 06, 08, 00, 00, 00, 00, 00, 00, 00, 00, 00, 02, 09, 02, 
            08, 00, 01, 00, 00, 00, 00, 00, 00, 00, 02, 12, 00, 00, 00, 02, 
            10, 12, 00, 00, 00, 00, 04, 00, 00, 00, 00, 01, 00, 00, 00, 06, 
            12, 01, 00, 00, 00, 02, 09, 00, 00, 00, 00, 00, 00, 00, 00, 03, 
            09, 00, 00, 00, 00, 04, 00, 00, 00, 00, 00, 00, 00, 00, 00, 02,
            08, 00, 00, 00, 00, 03, 08, 04, 04, 00, 04, 00, 00, 06, 08, 02,
            08, 00, 00, 00, 00, 00, 02, 09, 03, 08, 03, 08, 00, 01, 00, 02,
            08, 00, 00, 04, 00, 00, 02, 12, 06, 08, 00, 00, 00, 00, 04, 02,
            08, 00, 00, 03, 08, 00, 00, 01, 01, 00, 00, 00, 00, 02, 09, 02,
            08, 00, 00, 00, 00, 00, 00, 00, 00, 00, 02, 12, 00, 00, 00, 06,
            10, 12, 00, 00, 00, 00, 00, 00, 00, 04, 00, 01, 00, 00, 00, 03,
            08, 01, 00, 00, 00, 00, 06, 08, 00, 03, 08, 00, 00, 00, 00, 02,
            12, 00, 04, 00, 00, 00, 01, 00, 00, 00, 00, 00, 00, 00, 00, 02,
            09, 02, 09, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 06, 08, 02,
            12, 04, 04, 04, 04, 06, 12, 04, 04, 04, 06, 12, 04, 05, 04, 06,
        ];
        let board = raw_board.map(|x|{ return SquareFlags::from_bits(x).unwrap() });
        // 25 move bot position: [43, 226, 48, 18]
        // Changed to make running the test tractable
        let mut solver = IterativeDeepeningSolver::with_values(board,[191, 226, 177, 64], 201);
        let solution = solver.solve();
        assert_eq!(solution.len(), 10);
        assert_eq!(solution.last().unwrap().1, 201);
    }
}
