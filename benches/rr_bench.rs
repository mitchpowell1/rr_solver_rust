use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rr_solver::{IterativeDeepeningSolver, SquareFlags};

pub fn criterion_benchmark(crit: &mut Criterion) {
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
    ].map(|v| SquareFlags::from_bits(v).unwrap());
    let bots: [u8; 4] = [191, 226, 48, 16];
    let target = 201;
    crit.bench_function("medium_test", |b| b.iter(||{
        let mut solver = IterativeDeepeningSolver::with_values(raw_board, bots, target);
        black_box(solver.solve());
    }));
}

criterion_group!{
    name = benches; 
    config = Criterion::default().sample_size(500);
    targets = criterion_benchmark
}
criterion_main!(benches);
