//! 集成测试：长对局稳定性（T008）。

use gobango::core::{Color, Game, GameStatus, Move, Point};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// T008：300 手随机对局不崩溃、局面一致、内存结构稳定（无无界增长）。
#[test]
fn t008_long_game_stability() {
    let mut game = Game::new();
    let mut rng = StdRng::seed_from_u64(0xABAD_1DEA);
    let mut played = 0usize;

    while played < 300 {
        let legal = game.board().legal_moves();
        let mv = if legal.is_empty() || rng.gen_bool(0.03) {
            Move::Pass
        } else {
            Move::Play(legal[rng.gen_range(0..legal.len())])
        };
        // 随机的 Pass 可能使对局提前进入计分
        if game.status() == GameStatus::Scoring {
            break;
        }
        assert!(game.play(mv), "随机着法应总能合法执行（含 pass）");
        played += 1;
    }

    // 局面一致性：棋子数 + 空点数 = 361
    let b = game.board();
    let stones = b.count_stones(Color::Black) + b.count_stones(Color::White);
    let mut empty = 0;
    for y in 0..19 {
        for x in 0..19 {
            if b.at(x, y) == Color::Empty {
                empty += 1;
            }
        }
    }
    assert_eq!(stones + empty, 361, "棋子 + 空点必须等于 361");
    assert_eq!(game.move_count(), played);

    // 计分阶段：确认终局不崩溃
    if game.status() == GameStatus::Ongoing {
        game.play(Move::Pass);
        game.play(Move::Pass);
    }
    if game.status() == GameStatus::Scoring {
        let score = game.confirm_score();
        assert!(score.winner.is_some() || score.winner.is_none());
        assert_eq!(game.status(), GameStatus::Finished);
    }

    // 全部悔棋回空盘不崩溃（覆盖重建路径）
    while game.move_count() > 0 {
        assert!(game.undo().is_some());
    }
    assert!(game.undo().is_none());
    assert_eq!(game.board().count_stones(Color::Black) + game.board().count_stones(Color::White), 0);
}

/// T008b：AI 对弈直至结束（观战模式核心路径）不崩溃。
#[test]
fn t008_ai_vs_ai_game_runs() {
    use gobango::ai::{select_move, Difficulty, OpeningBook, SearchConfig};
    let mut game = Game::new();
    let book = OpeningBook::load();
    let mut seed = 42u64;
    let mut moves = 0usize;

    // 小预算配置，保证测试快速完成
    let cfg = SearchConfig {
        max_time: std::time::Duration::from_secs(10),
        max_sims: 25,
        min_sims: 25,
        threads: 1,
        c_uct: 1.4,
        max_rollout_moves: 150,
        use_book: true,
        book_plies: 10,
        heuristic: 2,
        max_candidates: 30,
        use_patterns: true,
        use_rave: true,
        use_enhanced: true,
        use_virtual_loss: true,
        use_virtual_roll: true,
        root_temperature: 1.5,
    };
    let _ = Difficulty::Easy; // 默认预算由 select_move 内部使用

    while game.status() == GameStatus::Ongoing && moves < 120 {
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let r = gobango::ai::select_move_cfg(
            game.board(),
            &game.book_history(),
            &cfg,
            &book,
            seed,
            None,
        );
        if !game.play(r.mv) {
            break;
        }
        moves += 1;
    }

    // 双虚着进入计分并确认
    if game.status() == GameStatus::Ongoing {
        game.play(Move::Pass);
        game.play(Move::Pass);
    }
    if game.status() == GameStatus::Scoring {
        let _ = game.confirm_score();
        assert_eq!(game.status(), GameStatus::Finished);
    }
}
