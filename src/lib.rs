mod iterative_deepening_solver;
mod util;

pub use iterative_deepening_solver::IterativeDeepeningSolver;
pub use util::SquareFlags;
use wasm_bindgen::prelude::*;


#[wasm_bindgen]
pub fn solve(board: js_sys::Array, bots: js_sys::Array, target: u8) -> js_sys::Array {
    let mut game_board: [util::SquareFlags; util::BOARD_SIZE] = [util::SquareFlags::empty(); util::BOARD_SIZE];
    let mut game_bots: [u8; util::BOT_COUNT] = [0; util::BOT_COUNT];
    board
        .iter()
        .enumerate()
        .for_each(|(i,v)|{game_board[i] = util::SquareFlags::from_bits(v.as_f64().unwrap() as u8).unwrap()});
    bots
        .iter()
        .enumerate()
        .for_each(|(i,v)|{game_bots[i] = v.as_f64().unwrap() as u8});
    let mut solver = IterativeDeepeningSolver::with_values(game_board, game_bots, target);
    let res = solver.solve();
    let out = js_sys::Array::new_with_length(res.len() as u32);
    for (i, v) in res.iter().enumerate() {
        let out_val: js_sys::Array = js_sys::Array::new_with_length(2);
        out_val.set(0, JsValue::from(v.0));
        out_val.set(1, JsValue::from(v.1));
        out.set(i as u32, JsValue::from(out_val));
    }
    out
}