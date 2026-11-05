//! 命令行对弈：人机 / 双人 / AI 观战三种模式。
//!
//! 交互：输入 `Q16` 落子、`pass` 虚着、`resign` 认输、`undo` 悔棋、`score` 查看当前计分。

use crate::ai::{select_move, Difficulty, OpeningBook};
use crate::core::{Color, Game, GameStatus, Move, Point};
use crate::nn_ai::{select_move_nn, AiEngine, NnDifficulty};
use std::io::{BufRead, Write};

/// 命令行模式。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CliMode {
    HumanVsHuman,
    HumanVsAi,
    AiVsAi,
}

/// 命令行参数。
#[derive(Clone, Debug)]
pub struct CliOptions {
    pub mode: CliMode,
    pub difficulty: Difficulty,
    pub human_color: Color,
    /// AI 引擎：经典或神经网络。
    pub engine: AiEngine,
    /// 神经网络难度（三档），仅 `engine == Neural` 时生效。
    pub nn_difficulty: NnDifficulty,
}

impl CliOptions {
    /// 当前档位的可读描述（按引擎取）。
    pub fn difficulty_text(&self) -> (&'static str, &'static str) {
        match self.engine {
            AiEngine::Classic => (self.difficulty.label(), self.difficulty.description()),
            AiEngine::Neural => (self.nn_difficulty.label(), self.nn_difficulty.description()),
        }
    }
}

/// 渲染文本棋盘。
pub fn render_board(game: &Game) -> String {
    let b = game.board();
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

/// 运行命令行对弈。
pub fn run_cli(opts: &CliOptions) -> std::io::Result<()> {
    let mut game = Game::new();
    let book = OpeningBook::load();
    let mut seed = 0xC11_2026u64;
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    println!("==== GobanGo 围棋 · 命令行对弈 ====");
    let (diff_label, diff_desc) = opts.difficulty_text();
    println!(
        "模式: {} | 引擎: {} | 难度: {} ({})",
        match opts.mode {
            CliMode::HumanVsHuman => "双人对弈",
            CliMode::HumanVsAi => match opts.human_color {
                Color::Black => "人机（玩家执黑）",
                _ => "人机（玩家执白）",
            },
            CliMode::AiVsAi => "AI 观战",
        },
        opts.engine.label(),
        diff_label,
        diff_desc
    );
    println!("输入坐标如 Q16 / pass / resign / undo / score 进行操作");

    loop {
        // 打印局面与状态
        print!("{}", render_board(&game));
        let status_line = match game.status() {
            GameStatus::Finished => format!(
                "对局结束，胜者：{}",
                game.winner().map(|c| c.label()).unwrap_or("和棋")
            ),
            GameStatus::Scoring => {
                let s = game.current_score();
                format!(
                    "计分阶段：黑 {} 子 / 白 {} 子（自动判死 {} 子）",
                    s.black_total, s.white_total, s.dead_removed
                )
            }
            GameStatus::Ongoing => format!(
                "第 {} 手 · {}行棋 · 黑提 {} / 白提 {}",
                game.move_count() + 1,
                game.current_player().label(),
                game.board().black_captures(),
                game.board().white_captures()
            ),
        };
        println!("{status_line}");

        if game.status() == GameStatus::Finished {
            let s = game.current_score();
            match s.winner {
                Some(Color::Black) => println!("黑胜 {} 子（黑 {} - 贴 {:.2}）", s.margin, s.black_total, game.komi()),
                Some(Color::White) => println!("白胜 {} 子（白 {} 子）", s.margin, s.white_total),
                _ => println!("和棋"),
            }
            break;
        }
        if game.status() == GameStatus::Scoring {
            println!("双方连续虚着，确认计分：");
            let s = game.confirm_score();
            match s.winner {
                Some(Color::Black) => println!("黑胜 {} 子", s.margin),
                Some(Color::White) => println!("白胜 {} 子", s.margin),
                _ => println!("和棋"),
            }
            continue;
        }

        // 决定本轮行动方
        let is_ai_turn = match opts.mode {
            CliMode::HumanVsAi => game.current_player() != opts.human_color,
            CliMode::AiVsAi => true,
            CliMode::HumanVsHuman => false,
        };

        if is_ai_turn {
            seed = seed.wrapping_add(0x9E3779B97F4A7C15);
            let r = match opts.engine {
                AiEngine::Classic => select_move(
                    game.board(),
                    &game.book_history(),
                    opts.difficulty,
                    &book,
                    seed,
                ),
                AiEngine::Neural => {
                    let snaps = game.recent_snapshots(crate::nn::HISTORY);
                    select_move_nn(&snaps, opts.nn_difficulty, seed).unwrap_or_else(|| {
                        // 模型不可用 → 回退经典引擎，保证对局不中断
                        select_move(
                            game.board(),
                            &game.book_history(),
                            Difficulty::Medium,
                            &book,
                            seed,
                        )
                    })
                }
            };
            match r.mv {
                Move::Play(p) => {
                    println!("电脑落子：{}（搜索 {:.2}s，{} 次模拟）", p.sgf(), r.elapsed.as_secs_f64(), r.total_sims);
                    game.play(r.mv);
                }
                Move::Pass => {
                    println!("电脑虚着");
                    game.play(Move::Pass);
                }
                Move::Resign => {
                    println!("电脑认输");
                    game.play(Move::Resign);
                }
            }
            if opts.mode == CliMode::AiVsAi {
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            continue;
        }

        // 人类输入
        let mut line = String::new();
        print!("你执{}，请落子> ", game.current_player().label());
        stdout.flush()?;
        line.clear();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let input = line.trim();
        match input.to_ascii_lowercase().as_str() {
            "pass" => {
                game.play(Move::Pass);
            }
            "resign" => {
                game.play(Move::Resign);
            }
            "undo" => {
                if opts.mode == CliMode::HumanVsAi {
                    game.undo();
                    game.undo(); // 撤销 AI 一手 + 人类一手
                } else {
                    game.undo();
                }
            }
            "score" => {
                let s = game.current_score();
                println!("黑 {} 子（{}+{}）｜白 {} 子（{}+{}）",
                    s.black_total, s.black_stones, s.black_territory,
                    s.white_total, s.white_stones, s.white_territory);
            }
            "help" => println!("Q16 / pass / resign / undo / score / help"),
            _ => match Point::from_sgf(input) {
                Some(p) => {
                    if !game.play(Move::Play(p)) {
                        println!("非法着法：{}", p.sgf());
                    }
                }
                None => println!("无法识别的输入：{}（参考 Q16 / pass / resign / undo）", input),
            },
        }
    }
    Ok(())
}
