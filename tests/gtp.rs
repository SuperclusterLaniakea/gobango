//! 集成测试：GTP 协议序列（T006）。

use gobango::core::Point;
use gobango::gtp::{run_commands, GtpEngine};
use std::time::{Duration, Instant};

/// T006a：GTP 命令序列基本流程。
#[test]
fn t006_gtp_command_sequence() {
    let mut engine = GtpEngine::new();
    let resp = run_commands(
        &mut engine,
        &[
            "boardsize 19",
            "clear_board",
            "komi 7.5",
            "play B Q16",
            "play W D4",
            "play B R4",
            "undo",
            "genmove B",
            "known_command genmove",
            "list_commands",
            "final_score",
        ],
    );

    // 前 7 条均为成功响应
    for r in &resp[..7] {
        assert!(r.starts_with("= "), "应成功响应，得到: {r:?}");
    }

    // genmove 返回合法着法
    let gen = resp[7].trim();
    assert!(gen.starts_with("= "), "genmove 应成功: {gen:?}");
    let mv = gen.trim_start_matches("= ").trim();
    assert!(
        mv == "pass" || mv == "resign" || Point::from_sgf(mv).is_some(),
        "genmove 返回了非法坐标: {mv}"
    );
    // 引擎内部局面一致：genmove B 之后轮到白
    assert_eq!(engine.game().current_player(), gobango::core::Color::White);

    // known_command
    assert!(resp[8].starts_with("= true"), "known_command genmove 应为 true");
    // list_commands 应包含 genmove
    assert!(resp[9].contains("genmove"));
    // final_score 应有结果格式
    assert!(resp[10].starts_with("= "));
    let score = resp[10].trim_start_matches("= ").trim();
    assert!(
        score == "0" || score.starts_with("B+") || score.starts_with("W+"),
        "final_score 格式异常: {score}"
    );
}

/// T006b：genmove 在时限内返回合法着法。
#[test]
fn t006_genmove_within_time() {
    let mut engine = GtpEngine::new();
    run_commands(
        &mut engine,
        &["boardsize 19", "clear_board", "play B Q16", "play W D4"],
    );
    let t0 = Instant::now();
    let r = engine.handle("genmove B");
    let elapsed = t0.elapsed();
    assert!(r.starts_with("= "), "genmove 失败: {r}");
    assert!(
        elapsed < Duration::from_secs(5),
        "genmove 应 ≤5s 返回，实际 {elapsed:?}"
    );
    let mv = r.trim_start_matches("= ").trim();
    if mv != "pass" && mv != "resign" {
        assert!(Point::from_sgf(mv).is_some());
    }
}

/// T006c：错误命令与非法着法返回 `?` 错误。
#[test]
fn t006_gtp_error_responses() {
    let mut engine = GtpEngine::new();
    let resp = run_commands(
        &mut engine,
        &[
            "boardsize 9",     // 只支持 19
            "play B I5",       // I 列不存在
            "play W A1",       // 位置合法（A1 = 左上角）
            "play W A1",       // 重复落子 → 非法
            "bogus_command",   // 未知命令
        ],
    );
    assert!(resp[0].starts_with("? "), "boardsize 9 应被拒绝: {}", resp[0]);
    assert!(resp[1].starts_with("? "), "play B I5 应报错: {}", resp[1]);
    assert!(resp[2].starts_with("= "), "play W A1 应成功: {}", resp[2]);
    assert!(resp[3].starts_with("? "), "重复落子应报错: {}", resp[3]);
    assert!(resp[4].starts_with("? "), "未知命令应报错: {}", resp[4]);
}
