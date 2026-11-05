//! SGF 棋谱模块：整盘棋谱的保存与解析（用于复盘与棋谱文件交换）。
//!
//! 采用最小 SGF 子集：`(;GM[1]FF[4]SZ[19]KM[7.5]RE[...];B[qq];W[dd]...)`。
//! 坐标使用小写字母（a-t，跳过 i）；空坐标 `[]` 表示虚着。

use crate::core::{Color, Game, Move, Point};

/// 内部坐标 → SGF 小写点坐标（如 (15,15) → "qq"）。
pub fn pt_to_sgf(p: Point) -> String {
    let lx = if p.x < 8 {
        (b'a' + p.x as u8) as char
    } else {
        (b'a' + p.x as u8 + 1) as char
    };
    let ly = if p.y < 8 {
        (b'a' + p.y as u8) as char
    } else {
        (b'a' + p.y as u8 + 1) as char
    };
    format!("{lx}{ly}")
}

/// SGF 小写点坐标 → 内部坐标。
pub fn pt_from_sgf(s: &str) -> Option<Point> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let mut cx = b[0].to_ascii_lowercase();
    let mut cy = b[1].to_ascii_lowercase();
    if !(b'a'..=b't').contains(&cx) || !(b'a'..=b't').contains(&cy) {
        return None;
    }
    if cx == b'i' || cy == b'i' {
        return None;
    }
    if cx > b'i' {
        cx -= 1;
    }
    if cy > b'i' {
        cy -= 1;
    }
    let x = (cx - b'a') as usize;
    let y = (cy - b'a') as usize;
    if x >= 19 || y >= 19 {
        return None;
    }
    Some(Point::new(x, y))
}

/// 将对局序列化为 SGF 文本。
pub fn to_sgf(game: &Game) -> String {
    let mut out = String::from("(;GM[1]FF[4]CA[UTF-8]AP[GobanGo:1.0.0]SZ[19]");
    out.push_str(&format!("KM[{:.1}]", game.komi()));
    // 终局结果（如已计分/认输）
    if let Some(w) = game.winner() {
        let s = game.current_score();
        let re = match w {
            Color::Black => format!("B+{:.1}", s.margin),
            Color::White => format!("W+{:.1}", s.margin),
            Color::Empty => "0".to_string(),
        };
        out.push_str(&format!("RE[{}]", re));
    }
    for (m, c) in game.history() {
        match m {
            Move::Play(p) => out.push_str(&format!(";{}[{}]", c.sgf(), pt_to_sgf(*p))),
            Move::Pass => out.push_str(&format!(";{}[]", c.sgf())),
            Move::Resign => {} // 认输由 RE 记录
        }
    }
    out.push(')');
    out
}

/// 解析 SGF 文本中的着法序列（含行棋方）。
/// 遇到非法着法会跳过并继续（容错加载）。
pub fn parse_moves(text: &str) -> Result<Vec<(Color, Move)>, String> {
    let start = text
        .find("(;")
        .ok_or_else(|| "缺少根节点 (;".to_string())?;
    let body = &text[start + 2..];
    let mut moves = Vec::new();

    // 按 ';' 分段：第一段为根属性，其后每段是一个着法节点
    for seg in body.split(';').skip(1) {
        let bytes = seg.as_bytes();
        let mut mv: Option<(Color, Move)> = None;
        let mut i = 0usize;
        while i + 1 < bytes.len() {
            let c = bytes[i];
            if (c == b'B' || c == b'W') && bytes[i + 1] == b'[' {
                let color = if c == b'B' { Color::Black } else { Color::White };
                let val_start = i + 2;
                let mut j = val_start;
                while j < bytes.len() && bytes[j] != b']' {
                    j += 1;
                }
                let val = seg[val_start..j.min(seg.len())].trim();
                mv = if val.is_empty() {
                    Some((color, Move::Pass))
                } else if let Some(p) = pt_from_sgf(val) {
                    Some((color, Move::Play(p)))
                } else {
                    break; // 非法点坐标：停止解析本段
                };
                break;
            }
            i += 1;
        }
        if let Some(m) = mv {
            moves.push(m);
        }
    }
    Ok(moves)
}

/// 将 SGF 文本恢复为一局棋（容错：非法着法跳过）。
/// 返回重建的对局。
pub fn game_from_sgf(text: &str) -> Result<Game, String> {
    let moves = parse_moves(text)?;
    let mut game = Game::new();
    for (c, m) in moves {
        // 与记录的行棋方不一致时先虚着同步（容错）
        if game.current_player() != c {
            game.play(Move::Pass);
        }
        let ok = match m {
            Move::Play(p) => game.play(Move::Play(p)),
            Move::Pass => game.play(Move::Pass),
            Move::Resign => game.play(Move::Resign),
        };
        if !ok && !m.is_pass() {
            // 跳过非法着法（如提子后留下的历史矛盾），继续加载
        }
    }
    Ok(game)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgf_point_codec() {
        for (x, y) in [
            (0usize, 0usize),
            (3, 3),
            (8, 3),
            (15, 15),
            (18, 18),
            (16, 3),
            (9, 9),
        ] {
            let p = Point::new(x, y);
            let s = pt_to_sgf(p);
            assert_eq!(pt_from_sgf(&s), Some(p), "{x},{y} -> {s}");
        }
        assert_eq!(pt_from_sgf("ii"), None, "SGF 跳过 i 列/行");
        assert_eq!(pt_from_sgf("a"), None);
        assert_eq!(pt_from_sgf("zz"), None);
    }

    #[test]
    fn sgf_roundtrip() {
        let mut g = Game::new();
        for (x, y) in [(3u8, 3u8), (15, 15), (3, 15), (15, 3), (9, 9)] {
            g.play(Move::Play(Point::new(x as usize, y as usize)));
        }
        let sgf = to_sgf(&g);
        let g2 = game_from_sgf(&sgf).unwrap();
        assert_eq!(g2.move_count(), g.move_count());
        for y in 0..19 {
            for x in 0..19 {
                assert_eq!(g2.board().at(x, y), g.board().at(x, y), "({x},{y})");
            }
        }
    }

    #[test]
    fn sgf_with_pass_and_re() {
        let mut g = Game::new();
        g.play(Move::Play(Point::new(3, 3)));
        g.play(Move::Play(Point::new(15, 15)));
        g.play(Move::Pass);
        g.play(Move::Pass); // 进入计分
        let s = g.confirm_score();
        let _ = s;
        let sgf = to_sgf(&g);
        assert!(sgf.contains("RE["));
        let g2 = game_from_sgf(&sgf).unwrap();
        assert_eq!(g2.move_count(), 4);
        assert_eq!(g2.board().at(3, 3), Color::Black);
        assert_eq!(g2.board().at(15, 15), Color::White);
        // 双虚着后应进入计分阶段
        assert_eq!(g2.status(), crate::core::GameStatus::Scoring);
    }

    #[test]
    fn sgf_parse_handwritten() {
        let sgf = "(;GM[1]FF[4]SZ[19]KM[7.5]PB[GobanGo]PW[GobanGo];B[qq];W[dd];B[pd])";
        let moves = parse_moves(sgf).unwrap();
        assert_eq!(moves.len(), 3);
        assert_eq!(moves[0], (Color::Black, Move::Play(Point::new(15, 15))));
        assert_eq!(moves[1], (Color::White, Move::Play(Point::new(3, 3))));
        assert_eq!(moves[2].0, Color::Black);
        // 解析失败路径
        assert!(parse_moves("no root").is_err());
    }
}
