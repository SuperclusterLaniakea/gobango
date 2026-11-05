//! GobanGo 围棋程序入口。
//!
//! 用法：
//! ```text
//! gobango                 # 启动 GUI
//! gobango --gtp           # 启动 GTP 引擎（可接入 Sabaki / Lizzie）
//! gobango --mode human_ai --difficulty medium --color black   # 命令行人机对弈
//! gobango --mode ai_ai --difficulty hard     # 命令行 AI 观战
//! gobango --help          # 帮助
//! ```

use gobango::cli::{run_cli, CliMode, CliOptions};
use gobango::core::Color;
use gobango::ai::Difficulty;
use gobango::nn_ai::{AiEngine, NnDifficulty};

fn print_help() {
    println!("GobanGo 围棋 1.2.0 — 19 路棋盘、神经网络 AI、中国规则计分、GTP 协议");
    println!();
    println!("用法:");
    println!("  gobango                    启动图形界面 (GUI)");
    println!("  gobango --gtp              启动 GTP 引擎（接入 Sabaki / Lizzie）");
    println!("  gobango --mode human_ai [--engine nn|classic] [--nn-difficulty low|medium|high]");
    println!("  gobango --mode ai_ai   [--engine nn|classic] [--difficulty 档位]");
    println!("  gobango --mode human_human");
    println!("  gobango --help");
    println!();
    println!("AI 引擎:");
    println!("  nn      神经网络（默认）：近千万局棋谱训练的双头网络，三档难度");
    println!("  classic 经典引擎：MCTS + 启发式走子 + 开局库，四档难度");
    println!();
    println!("神经网络难度（--nn-difficulty）:");
    println!("  低: 纯策略快速出招，随机性较大，适合初学者");
    println!("  中: 纯策略直接选点，多走网络最看好处");
    println!("  高: 策略网络 + 树搜索（160 次模拟），会算清局部变化");
    println!();
    println!("经典难度（--difficulty）:");
    println!("  入门 / 中级 / 高级 / 大师");
}

fn parse_difficulty(s: &str) -> Option<Difficulty> {
    match s {
        "easy" | "入门" | "1" => Some(Difficulty::Easy),
        "medium" | "中级" | "2" => Some(Difficulty::Medium),
        "hard" | "高级" | "3" => Some(Difficulty::Hard),
        "master" | "大师" | "4" => Some(Difficulty::Master),
        _ => None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut gtp_mode = false;
    let mut mode = CliMode::HumanVsAi;
    let mut difficulty = Difficulty::Medium;
    let mut human_color = Color::Black;
    // 默认启用神经网络引擎（加载失败时各项入口会自动回退经典引擎）
    let mut engine = AiEngine::Neural;
    let mut nn_difficulty = NnDifficulty::Medium;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--gtp" => gtp_mode = true,
            "--mode" => {
                i += 1;
                if i >= args.len() {
                    print_help();
                    return Ok(());
                }
                mode = match args[i].as_str() {
                    "human_ai" | "human-ai" => CliMode::HumanVsAi,
                    "ai_ai" | "ai-ai" | "spectate" => CliMode::AiVsAi,
                    "human_human" | "human-human" | "hotseat" => CliMode::HumanVsHuman,
                    _ => {
                        print_help();
                        return Ok(());
                    }
                };
            }
            "--difficulty" => {
                i += 1;
                if i >= args.len() {
                    print_help();
                    return Ok(());
                }
                match parse_difficulty(&args[i]) {
                    Some(d) => difficulty = d,
                    None => {
                        print_help();
                        return Ok(());
                    }
                }
            }
            "--color" => {
                i += 1;
                if i >= args.len() {
                    print_help();
                    return Ok(());
                }
                human_color = match args[i].as_str() {
                    "black" | "b" | "B" => Color::Black,
                    "white" | "w" | "W" => Color::White,
                    _ => {
                        print_help();
                        return Ok(());
                    }
                };
            }
            "--engine" => {
                i += 1;
                if i >= args.len() {
                    print_help();
                    return Ok(());
                }
                match AiEngine::parse(&args[i]) {
                    Some(e) => engine = e,
                    None => {
                        println!("未知引擎: {}", args[i]);
                        print_help();
                        return Ok(());
                    }
                }
            }
            "--nn-difficulty" | "--nn" => {
                i += 1;
                if i >= args.len() {
                    print_help();
                    return Ok(());
                }
                match NnDifficulty::parse(&args[i]) {
                    Some(d) => nn_difficulty = d,
                    None => {
                        println!("未知神经网络难度: {}", args[i]);
                        print_help();
                        return Ok(());
                    }
                }
            }
            other => {
                println!("未知参数: {other}\n");
                print_help();
                return Ok(());
            }
        }
        i += 1;
    }

    if gtp_mode {
        // GTP 引擎默认使用神经网络，模型加载失败时自动回退经典引擎
        if !gobango::nn::init(None) {
            eprintln!("[提示] 神经网络模型加载失败，GTP 引擎将使用经典引擎");
        }
        gobango::gtp::run_gtp()?;
        return Ok(());
    }

    // 初始化神经网络（内嵌模型；失败则各入口自动回退经典引擎）
    if engine == AiEngine::Neural && !gobango::nn::init(None) {
        eprintln!("[提示] 神经网络模型加载失败，本次将使用经典引擎");
    }

    if mode == CliMode::HumanVsAi
        || mode == CliMode::AiVsAi
        || mode == CliMode::HumanVsHuman
    {
        // 显式指定了 --mode 才走命令行
        let had_mode = args.iter().any(|a| a == "--mode");
        if had_mode {
            let opts = CliOptions {
                mode,
                difficulty,
                human_color,
                engine,
                nn_difficulty,
            };
            run_cli(&opts)?;
            return Ok(());
        }
    }

    // 默认：GUI
    gui::run_gui()?;
    Ok(())
}

// 顶层 re-export，供 main 直接调用
mod gui {
    pub fn run_gui() -> eframe::Result<()> {
        gobango::gui::run_gui()
    }
}
