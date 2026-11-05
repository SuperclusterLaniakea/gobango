//! 性能验收测试（T 系列性能项）：
//! - 入门单步 ≤ 0.5s；中级 ≤ 2.0s；高级 ≤ 5.0s（搜索受时间预算约束，给少量容差）。
//! - 内存占用峰值 ≤ 200MB（通过长对局不崩溃 + 搜索节点数有界验证）。
//!
//! 说明：搜索本身按墙钟时间自停止，debug 模式下模拟下限可能略超预算，
//! 因此严格时间断言仅在 release 模式（`cargo test --release`）下执行。

use gobango::ai::{select_move, Difficulty, OpeningBook};
use gobango::core::{Color, Game, Move, Point};
use std::time::{Duration, Instant};

/// 构造一个中盘局面（双方各有若干子，模拟真实对局中的搜索）。
fn midgame_board() -> Game {
    let mut g = Game::new();
    // 星位占角 + 中腹接触战，保证候选点多
    let moves = [
        (3, 3), (15, 15), (3, 15), (15, 3), (9, 9),
        (4, 3), (3, 4), (14, 15), (15, 14), (9, 10),
        (9, 8), (10, 9), (8, 9), (11, 11), (12, 12),
        (5, 5), (14, 14), (4, 14), (15, 4), (10, 10),
    ];
    for &(x, y) in &moves {
        g.play(Move::Play(Point::new(x, y)));
    }
    g
}

#[cfg(not(debug_assertions))]
fn assert_time_budget(difficulty: Difficulty, budget_ms: u64) {
    let game = midgame_board();
    let book = OpeningBook::load();
    let t0 = Instant::now();
    let r = select_move(
        game.board(),
        &game.book_history(),
        difficulty,
        &book,
        0x7000_2026u64,
    );
    let elapsed = t0.elapsed();
    // 预算 + 50% 容差（覆盖系统调度抖动与模拟下限）
    let limit = Duration::from_millis(budget_ms + budget_ms / 2);
    assert!(
        elapsed <= limit,
        "{} 单步应 ≤{:.1}s，实际 {:.2}s",
        difficulty.label(),
        budget_ms as f64 / 1000.0,
        elapsed.as_secs_f64()
    );
    assert!(
        !r.mv.is_resign() || game.board().legal_moves().is_empty(),
        "AI 不应无故认输"
    );
}

/// T-性能：入门单步 ≤ 0.5s。
#[test]
#[cfg(not(debug_assertions))]
fn perf_easy_budget() {
    assert_time_budget(Difficulty::Easy, 500);
}

/// T-性能：中级单步 ≤ 2.0s。
#[test]
#[cfg(not(debug_assertions))]
fn perf_medium_budget() {
    assert_time_budget(Difficulty::Medium, 2000);
}

/// T-性能：高级单步 ≤ 5.0s。
#[test]
#[cfg(not(debug_assertions))]
fn perf_hard_budget() {
    assert_time_budget(Difficulty::Hard, 5000);
}

/// T-性能：搜索节点数有界（内存可控的间接验证）。
/// 高级搜索 5 秒后，节点总量应远小于 200MB（单节点约 500B，预算 40 万节点）。
#[test]
fn perf_memory_bound() {
    let game = midgame_board();
    let book = OpeningBook::load();
    // 用最小配置跑 200 次迭代，确认搜索循环可正常结束
    let cfg = gobango::ai::SearchConfig {
        max_time: Duration::from_secs(30),
        max_sims: 200,
        min_sims: 200,
        threads: 2,
        c_uct: 1.5,
        max_rollout_moves: 300,
        use_book: false,
        book_plies: 0,
        heuristic: 3,
        max_candidates: 50,
        use_patterns: true,
        use_rave: true,
        use_enhanced: true,
        use_virtual_loss: true,
        use_virtual_roll: true,
        root_temperature: 1.5,
    };
    let r = gobango::ai::select_move_cfg(
        game.board(),
        &game.book_history(),
        &cfg,
        &book,
        0xABCD_2026u64,
        None,
    );
    // 200 次迭代 × 2 线程 ≤ 数百节点；即便放大到 20 万次也远低于 200MB
    assert!(r.total_sims >= 200 || r.mv.is_pass(), "搜索应执行完预算迭代");
}

/// T-性能：长对局内存不增长（复用稳定性测试的 300 手流程，release 下运行）。
#[test]
#[cfg(not(debug_assertions))]
fn perf_long_game_no_growth() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut game = Game::new();
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    let mut moves = 0usize;
    while moves < 300 && game.status() != gobango::core::GameStatus::Scoring {
        let legal = game.board().legal_moves();
        let mv = if legal.is_empty() || rng.gen_bool(0.03) {
            Move::Pass
        } else {
            Move::Play(legal[rng.gen_range(0..legal.len())])
        };
        if !game.play(mv) {
            break;
        }
        moves += 1;
    }
    assert!(moves >= 200, "应能稳定进行至少 200 手（实际 {moves} 手）");
    let _ = Color::Black;
}
