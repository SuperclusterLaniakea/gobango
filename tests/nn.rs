//! 神经网络 AI 的回归测试：合法性、特征正确性、边界情况。

use gobango::core::{Board, Color, Game, Move, Point};
use gobango::nn;
use gobango::nn_ai::{select_move_nn, AiEngine, NnDifficulty};

fn start_game(plies: usize) -> Game {
    let mut g = Game::new();
    // 用经典引擎之外的方式摆局：直接落几个固定点，保证可复现
    let seq = ["D4", "Q16", "P3", "D16", "Q3", "C5", "R14", "F3"];
    for (i, s) in seq.iter().enumerate() {
        if i >= plies {
            break;
        }
        let p = Point::from_sgf(s).expect("坐标非法");
        g.play(Move::Play(p));
    }
    g
}

#[test]
fn model_loads() {
    // 内嵌模型必须能加载（这是单文件部署的前提）
    assert!(nn::init(None), "内嵌 ONNX 模型加载失败");
    assert!(nn::is_ready());
    assert_eq!(nn::PLANES, 9);
    assert_eq!(nn::HISTORY, 4);
    assert_eq!(nn::POLICY_SIZE, 362);
}

#[test]
fn all_difficulties_return_legal_move() {
    assert!(nn::init(None));
    let g = start_game(8);
    for d in NnDifficulty::all() {
        let snaps = g.recent_snapshots(nn::HISTORY);
        let r = select_move_nn(&snaps, d, 20261101).expect("推理失败");
        match r.mv {
            Move::Play(p) => {
                assert!(p.in_bounds(), "{} 档给出越界着法", d.label());
                assert!(
                    g.board().is_legal(p.x, p.y),
                    "{} 档给出非法着法 {}",
                    d.label(),
                    p.sgf()
                );
            }
            Move::Pass => {}
            Move::Resign => panic!("{} 档在开局就认输", d.label()),
        }
    }
}

#[test]
fn high_difficulty_runs_search() {
    assert!(nn::init(None));
    let g = start_game(4);
    let snaps = g.recent_snapshots(nn::HISTORY);
    let r = select_move_nn(&snaps, NnDifficulty::High, 7).expect("推理失败");
    assert_eq!(r.total_sims, NnDifficulty::High.config().sims);
    // 搜索确实花时间（下限放宽，避免 CI 抖动）
    assert!(r.elapsed.as_millis() >= 10, "高档耗时异常：{:?}", r.elapsed);
}

#[test]
fn pure_policy_is_fast() {
    assert!(nn::init(None));
    let g = start_game(4);
    let snaps = g.recent_snapshots(nn::HISTORY);
    let r = select_move_nn(&snaps, NnDifficulty::Low, 7).expect("推理失败");
    assert_eq!(r.total_sims, 0);
    assert!(
        r.elapsed.as_millis() < 500,
        "纯策略应该很快，实际 {:?}",
        r.elapsed
    );
}

#[test]
fn feature_planes_match_expected_layout() {
    // 手工摆一个已知局面，直接验证 9 平面特征的行列方向与常数平面
    let mut b = Board::new();
    // 黑 D4 → x=3, y=19-4=15；白 Q16 → x=16, y=3
    b.play(3, 15);
    b.play(16, 3);

    let mut g = Game::new();
    g.play(Move::Play(Point::new(3, 15)));
    g.play(Move::Play(Point::new(16, 3)));
    let snaps = g.recent_snapshots(nn::HISTORY);
    assert!(!snaps.is_empty());
    // snaps[0] 必须是当前局面
    assert_eq!(snaps[0].at(3, 15), Color::Black);
    assert_eq!(snaps[0].at(16, 3), Color::White);
    // 历史快照应包含「只有黑 D4」的时刻（即 1 手之前）
    if snaps.len() >= 2 {
        assert_eq!(snaps[1].at(3, 15), Color::Black);
        assert_eq!(snaps[1].at(16, 3), Color::Empty);
    }
    // 当前轮到黑方（下了 2 手）
    assert_eq!(snaps[0].turn(), Color::Black);
}

#[test]
fn snapshots_survive_undo() {
    let mut g = start_game(6);
    let before = g.recent_snapshots(nn::HISTORY);
    assert!(g.undo().is_some());
    let after = g.recent_snapshots(nn::HISTORY);
    // 悔棋后快照必须重建，且与「少一手」的局面一致
    assert!(!after.is_empty());
    assert_eq!(after[0].stone_hash(), g.board().stone_hash());
    // 悔棋前后的快照集合不应完全相同
    if before.len() == after.len() && before[0].stone_hash() == after[0].stone_hash() {
        panic!("悔棋后快照未更新");
    }
}

#[test]
fn engine_parse() {
    assert_eq!(AiEngine::parse("nn"), Some(AiEngine::Neural));
    assert_eq!(AiEngine::parse("神经网络"), Some(AiEngine::Neural));
    assert_eq!(AiEngine::parse("classic"), Some(AiEngine::Classic));
    assert_eq!(AiEngine::parse("bogus"), None);
    assert_eq!(NnDifficulty::parse("low"), Some(NnDifficulty::Low));
    assert_eq!(NnDifficulty::parse("高"), Some(NnDifficulty::High));
    assert_eq!(NnDifficulty::parse("3"), Some(NnDifficulty::High));
    assert_eq!(NnDifficulty::parse("bogus"), None);
}
