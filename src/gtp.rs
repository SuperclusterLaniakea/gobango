//! GTP 2.0 协议引擎：可接入 Sabaki / Lizzie 等围棋客户端。
//!
//! 支持命令子集：boardsize、clear_board、komi、play、genmove、undo、showboard、
//! final_score、list_commands、known_command、time_settings、goban_difficulty、
//! name、version、protocol_version、quit。

use crate::ai::{select_move, Difficulty, OpeningBook};
use crate::core::{compute_score, Color, Game, GameStatus, Move, Point};
use crate::nn_ai::{select_move_nn, AiEngine, NnDifficulty};
use std::io::{BufRead, Write};

/// GTP 引擎状态。
pub struct GtpEngine {
    game: Game,
    difficulty: Difficulty,
    /// AI 引擎：经典或神经网络。
    engine: AiEngine,
    /// 神经网络三档难度。
    nn_difficulty: NnDifficulty,
    book: OpeningBook,
    seed: u64,
    quit: bool,
}

impl Default for GtpEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl GtpEngine {
    pub fn new() -> Self {
        GtpEngine {
            game: Game::new(),
            difficulty: Difficulty::Medium,
            engine: AiEngine::Neural,
            nn_difficulty: NnDifficulty::Medium,
            book: OpeningBook::load(),
            seed: 0x5EED_2026,
            quit: false,
        }
    }

    /// 当前对局（供测试与调试使用）。
    pub fn game(&self) -> &Game {
        &self.game
    }

    /// 处理一行 GTP 命令，返回完整响应（含 `=`/`?` 前缀）。
    pub fn handle(&mut self, line: &str) -> String {
        let line = line.trim();
        if line.is_empty() {
            return String::new();
        }
        // 解析可选命令 id
        let (id, rest) = match line.find(char::is_whitespace) {
            Some(i) if line[..i].chars().all(|c| c.is_ascii_digit()) => {
                (&line[..i], line[i..].trim())
            }
            _ => ("", line),
        };
        let prefix = |ok: bool| if ok { format!("={}", id) } else { format!("?{}", id) };
        let mut parts = rest.split_whitespace();
        let cmd = parts.next().unwrap_or("").to_ascii_lowercase();
        let args: Vec<&str> = parts.collect();

        let result = match cmd.as_str() {
            "boardsize" => {
                if args.first().map_or(false, |s| *s == "19") {
                    self.game.reset();
                    "".to_string()
                } else {
                    return format!("{} unacceptable size (only 19 supported)", prefix(false));
                }
            }
            "clear_board" => {
                self.game.reset();
                "".to_string()
            }
            "komi" => match args.first().and_then(|s| s.parse::<f64>().ok()) {
                Some(k) => {
                    self.game.set_komi(k);
                    "".to_string()
                }
                None => return format!("{} invalid komi", prefix(false)),
            },
            "play" => {
                let color = parse_color(args.first().copied().unwrap_or(""));
                let mv = parse_move(args.get(1).copied().unwrap_or(""));
                if color == Color::Empty || mv.is_none() {
                    return format!("{} invalid play command", prefix(false));
                }
                let mv = mv.unwrap();
                // 若与引擎记录的行棋方不一致，先让当前方虚着同步
                if self.game.current_player() != color {
                    self.game.play(Move::Pass);
                }
                if !self.game.play(mv) {
                    return format!("{} illegal move: {}", prefix(false), args.get(1).unwrap());
                }
                "".to_string()
            }
            "genmove" => {
                let color = parse_color(args.first().copied().unwrap_or(""));
                if color == Color::Empty {
                    return format!("{} invalid color", prefix(false));
                }
                if self.game.status() == GameStatus::Finished {
                    self.game.reset();
                }
                if self.game.current_player() != color {
                    self.game.play(Move::Pass);
                }
                self.seed = self.seed.wrapping_add(0x9E3779B97F4A7C15);
                let result = match self.engine {
                    AiEngine::Classic => select_move(
                        self.game.board(),
                        &self.game.book_history(),
                        self.difficulty,
                        &self.book,
                        self.seed,
                    ),
                    AiEngine::Neural => {
                        let snaps = self.game.recent_snapshots(crate::nn::HISTORY);
                        match select_move_nn(&snaps, self.nn_difficulty, self.seed) {
                            Some(r) => r,
                            // 模型不可用 → 回退经典引擎
                            None => select_move(
                                self.game.board(),
                                &self.game.book_history(),
                                self.difficulty,
                                &self.book,
                                self.seed,
                            ),
                        }
                    }
                };
                let mv = result.mv;
                if !self.game.play(mv) {
                    return format!("{} engine produced an illegal move", prefix(false));
                }
                match mv {
                    Move::Play(p) => p.sgf(),
                    Move::Pass => "pass".to_string(),
                    Move::Resign => "resign".to_string(),
                }
            }
            "undo" => {
                if self.game.undo().is_some() {
                    "".to_string()
                } else {
                    return format!("{} cannot undo", prefix(false));
                }
            }
            "showboard" => self.showboard(),
            "final_score" => {
                let score = compute_score(self.game.board(), self.game.dead());
                match score.winner {
                    Some(Color::Black) => format!("B+{:.1}", score.margin),
                    Some(Color::White) => format!("W+{:.1}", score.margin),
                    _ => "0".to_string(),
                }
            }
            "list_commands" => {
                "boardsize\nclear_board\nkomi\nplay\ngenmove\nundo\nshowboard\nfinal_score\nlist_commands\nknown_command\ntime_settings\ngoban_difficulty\ngobango_engine\nname\nversion\nprotocol_version\nquit"
                    .to_string()
            }
            "known_command" => {
                let known = [
                    "boardsize", "clear_board", "komi", "play", "genmove", "undo",
                    "showboard", "final_score", "list_commands", "known_command",
                    "time_settings", "goban_difficulty", "gobango_engine", "name", "version",
                    "protocol_version", "quit",
                ];
                let name = args.first().copied().unwrap_or("");
                if known.contains(&name) {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            "time_settings" => "".to_string(), // 接受但使用难度内建预算
            "goban_difficulty" => {
                // 约定：low/medium/high 走神经网络三档；easy/hard/master 走经典四档。
                // 这样客户端只配一个难度字符串就能切到对应引擎。
                let arg = args.first().copied().unwrap_or("");
                if let Some(d) = NnDifficulty::parse(arg) {
                    self.engine = AiEngine::Neural;
                    self.nn_difficulty = d;
                } else {
                    match arg {
                        "easy" | "入门" | "1" => self.difficulty = Difficulty::Easy,
                        "classic-medium" | "中级" => self.difficulty = Difficulty::Medium,
                        "hard" | "高级" => self.difficulty = Difficulty::Hard,
                        "master" | "大师" => self.difficulty = Difficulty::Master,
                        _ => {
                            return format!(
                                "{} unknown difficulty (low/medium/high | easy/hard/master)",
                                prefix(false)
                            )
                        }
                    }
                    self.engine = AiEngine::Classic;
                }
                "".to_string()
            }
            // 显式切换引擎，不动当前难度档位
            "gobango_engine" => match AiEngine::parse(args.first().copied().unwrap_or("")) {
                Some(e) => {
                    self.engine = e;
                    "".to_string()
                }
                None => format!("{} unknown engine (classic/nn)", prefix(false)),
            },
            "name" => "GobanGo".to_string(),
            "version" => "1.2.0".to_string(),
            "protocol_version" => "2".to_string(),
            "quit" => {
                self.quit = true;
                "".to_string()
            }
            _ => return format!("{} unknown command: {}", prefix(false), cmd),
        };
        format!("{} {}", prefix(true), result)
    }

    /// 文本棋盘（GTP showboard 用）。
    fn showboard(&self) -> String {
        let b = self.game.board();
        let mut out = String::new();
        out.push_str("    A B C D E F G H J K L M N O P Q R S T\n");
        for y in 0..19 {
            out.push_str(&format!("{:2} ", 19 - y));
            for x in 0..19 {
                out.push(match b.at(x, y) {
                    Color::Black => 'X',
                    Color::White => 'O',
                    Color::Empty => '.',
                });
                out.push(' ');
            }
            out.push_str(&format!("{:2}\n", 19 - y));
        }
        out.push_str("    A B C D E F G H J K L M N O P Q R S T\n");
        out
    }
}

/// 解析 GTP 颜色："B"/"W"。
/// 解析 GTP 颜色："B"/"W" 及客户端常用的 "black"/"white"。
fn parse_color(s: &str) -> Color {
    match s.to_ascii_lowercase().as_str() {
        "b" | "black" => Color::Black,
        "w" | "white" => Color::White,
        _ => Color::Empty,
    }
}

/// 解析 GTP 着法："D4" / "pass" / "resign"。
fn parse_move(s: &str) -> Option<Move> {
    match s.to_ascii_lowercase().as_str() {
        "pass" => Some(Move::Pass),
        "resign" => Some(Move::Resign),
        _ => Point::from_sgf(s).map(Move::Play),
    }
}

/// 运行 GTP 主循环（读 stdin，写 stdout）。
pub fn run_gtp() -> std::io::Result<()> {
    let mut engine = GtpEngine::new();
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut line = String::new();
    // 握手：协议版本
    writeln!(stdout, "= 2")?;
    stdout.flush()?;
    loop {
        line.clear();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let response = engine.handle(&line);
        if !response.is_empty() {
            writeln!(stdout, "{}", response)?;
            stdout.flush()?;
        }
        if engine.quit {
            break;
        }
    }
    Ok(())
}

/// 供测试使用：批量处理命令序列。
pub fn run_commands(engine: &mut GtpEngine, cmds: &[&str]) -> Vec<String> {
    cmds.iter().map(|c| engine.handle(c)).collect()
}
