//! GUI 模块：基于 egui 的美化界面。
//!
//! 功能：木纹棋盘、立体棋子、坐标与星位、最近一手标记、合法落点提示、
//! 悬停预览、死子标记（计分阶段）、三难度人机 / 双人 / AI 观战、
//! 中国规则计分展示、AI 后台线程搜索（不阻塞界面）。

use crate::ai::{select_move_cfg, Difficulty, OpeningBook, request_ai_cancel, clear_ai_cancel};
use crate::core::{compute_score, Color, Game, GameStatus, Move, Point, Score};
use crate::gpu::GpuEngine;
use crate::nn_ai::{select_move_nn, AiEngine, NnDifficulty};
use eframe::egui;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;
use std::time::Instant;

const BOARD_SIZE: usize = 19;
const STAR_POINTS: [(usize, usize); 9] = [
    (3, 3), (3, 9), (3, 15),
    (9, 3), (9, 9), (9, 15),
    (15, 3), (15, 9), (15, 15),
];

/// 对弈模式。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GuiMode {
    HumanVsHuman,
    HumanVsAi(Color),
    AiVsAi,
}

impl GuiMode {
    fn label(&self) -> &'static str {
        match self {
            GuiMode::HumanVsHuman => "双人对弈",
            GuiMode::HumanVsAi(Color::Black) => "人机 · 执黑",
            GuiMode::HumanVsAi(Color::White) => "人机 · 执白",
            GuiMode::HumanVsAi(_) => "人机 · 执黑",
            GuiMode::AiVsAi => "AI 观战",
        }
    }
}

/// AI 搜索线程返回的结果。
struct AiMoveResult {
    mv: Move,
    elapsed: std::time::Duration,
    sims: u32,
}

/// 主应用状态。
pub struct GoApp {
    game: Game,
    mode: GuiMode,
    difficulty: Difficulty,
    /// 当前 AI 引擎。
    ai_engine: AiEngine,
    /// 神经网络难度（三档）。
    nn_difficulty: NnDifficulty,
    message: String,
    /// 最近一手落点（用于标记）。
    last_played: Option<Point>,
    /// AI 是否正在思考。
    ai_busy: bool,
    ai_rx: Option<Receiver<AiMoveResult>>,
    ai_tx: Sender<AiMoveResult>,
    /// AI 搜索线程句柄（持有它才能在 GoApp 析构时 join，防止进程挂起）。
    ai_handle: Option<JoinHandle<()>>,
    /// 上一轮 AI 配置指纹（引擎 + 难度档位）：切换时主动取消上一轮搜索。
    last_ai_key: Option<(AiEngine, u8)>,
    /// 计分阶段是否处于“标记死子”模式。
    marking_dead: bool,
    seed: u64,
    /// 观战模式下相邻两步之间的最小间隔。
    last_ai_time: Instant,
    /// 悬停点（用于预览）。
    hover: Option<Point>,
    /// 合法点提示缓存（仅当局面哈希变化时重建）。
    hint_cache: Option<(u64, Vec<Point>)>,
    /// 手数映射（点 → 手数），用于棋子上显示编号。
    move_numbers: std::collections::HashMap<Point, usize>,
    /// 棋盘固定像素尺寸（0 = 自适应窗口）。
    board_px: f32,
    /// 复盘游标：None = 实时对局；Some(i) = 显示前 i 手后的局面。
    replay_index: Option<usize>,
    /// “判断胜负”的最近结果（仅展示，不改变对局状态）。
    judged_score: Option<Score>,
    /// 棋谱文件名输入框。
    save_name: String,
    /// 棋谱保存目录。
    games_dir: String,
    /// 已保存棋谱文件列表。
    sgf_files: Vec<String>,
    /// 下拉选中的棋谱文件。
    selected_sgf: Option<String>,
    /// 判断胜负后的确认对话框（Some = 等待用户确认是否结束）。
    confirm_dialog: Option<Score>,
    /// 终局胜利/失败特效。
    effect: Option<VictoryEffect>,
    /// GPU 计算引擎（None = CPU 计算；勾选后由 GPU 批量走子）。
    gpu_engine: Option<Arc<Mutex<GpuEngine>>>,
}

/// 终局特效状态。
struct VictoryEffect {
    /// 胜方（None = 和棋）。
    winner: Option<Color>,
    /// 人类执子颜色（人机模式下用于显示“胜利/失败”）。
    human_color: Option<Color>,
    start: Instant,
    duration: f32,
}

impl GoApp {
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        GoApp {
            game: Game::new(),
            mode: GuiMode::HumanVsAi(Color::Black),
            difficulty: Difficulty::Medium,
            ai_engine: AiEngine::Neural,
            nn_difficulty: NnDifficulty::Medium,
            message: "黑棋先行，点击棋盘落子".to_string(),
            last_played: None,
            ai_busy: false,
            ai_rx: Some(rx),
            ai_tx: tx,
            ai_handle: None,
            last_ai_key: None,
            marking_dead: false,
            seed: 0x6F0_2026,
            last_ai_time: Instant::now(),
            hover: None,
            hint_cache: None,
            move_numbers: std::collections::HashMap::new(),
            // 棋盘固定像素尺寸（0 = 自适应窗口；默认 600）
            board_px: 600.0,
            replay_index: None,
            judged_score: None,
            save_name: String::new(),
            games_dir: "games".to_string(),
            sgf_files: Vec::new(),
            selected_sgf: None,
            confirm_dialog: None,
            effect: None,
            gpu_engine: None,
        }
    }

    /// 新开一局。
    fn new_game(&mut self) {
        self.game.reset();
        self.last_played = None;
        self.marking_dead = false;
        self.replay_index = None;
        self.judged_score = None;
        self.confirm_dialog = None;
        self.effect = None;
        self.message = match self.mode {
            GuiMode::HumanVsHuman => "黑棋先行（双人对弈）".to_string(),
            GuiMode::HumanVsAi(Color::Black) => "黑棋先行（玩家执黑）".to_string(),
            GuiMode::HumanVsAi(Color::White) => "黑棋先行（电脑执黑）".to_string(),
            GuiMode::HumanVsAi(_) => "黑棋先行（人机对弈）".to_string(),
            GuiMode::AiVsAi => "AI 观战模式".to_string(),
        };
    }

    /// 当前是否轮到电脑行棋（复盘/确认对话框打开时不触发 AI）。
    fn is_ai_turn(&self) -> bool {
        if self.replay_index.is_some() || self.confirm_dialog.is_some() {
            return false;
        }
        if self.game.status() != GameStatus::Ongoing {
            return false;
        }
        match self.mode {
            GuiMode::HumanVsAi(human) => self.game.current_player() != human,
            GuiMode::AiVsAi => true,
            GuiMode::HumanVsHuman => false,
        }
    }

    /// 启动 AI 搜索线程（替换上一轮未完成的线程）。
    fn start_ai(&mut self) {
        // 先清理上一轮线程（可能被 Drop 取消，或难度已切换）
        let ai_key = (self.ai_engine, self.difficulty_index());
        if self.ai_busy || Some(ai_key) != self.last_ai_key {
            request_ai_cancel();
            if let Some(h) = self.ai_handle.take() {
                let _ = h.join();
            }
            self.ai_busy = false;
            self.last_ai_key = Some(ai_key);
        }
        if self.ai_busy {
            return;
        }
        let board = self.game.board().clone();
        let history = self.game.book_history();
        // 神经网络特征所需的局面快照（[0] = 当前局面）
        let snaps = self.game.recent_snapshots(crate::nn::HISTORY);
        let difficulty = self.difficulty;
        let nn_diff = self.nn_difficulty;
        let engine = self.ai_engine;
        self.seed = self.seed.wrapping_add(0x9E3779B97F4A7C15);
        let seed = self.seed;
        let tx = self.ai_tx.clone();
        let gpu = self.gpu_engine.clone();
        let gpu_on = gpu.is_some();
        self.ai_busy = true;
        self.message = format!(
            "{}思考中…（{} · {}）",
            self.game.current_player().label(),
            engine.label(),
            match engine {
                AiEngine::Classic => {
                    if gpu_on {
                        "GPU"
                    } else {
                        "CPU"
                    }
                }
                AiEngine::Neural => "策略网络",
            }
        );
        let handle = thread::spawn(move || {
            clear_ai_cancel();
            let r = match engine {
                AiEngine::Classic => {
                    let book = OpeningBook::load();
                    let cfg = difficulty.config();
                    select_move_cfg(&board, &history, &cfg, &book, seed, gpu)
                }
                AiEngine::Neural => match select_move_nn(&snaps, nn_diff, seed) {
                    Some(r) => r,
                    None => {
                        // 模型不可用 → 回退经典引擎，保证对局不中断
                        let book = OpeningBook::load();
                        let cfg = Difficulty::Medium.config();
                        select_move_cfg(&board, &history, &cfg, &book, seed, None)
                    }
                },
            };
            let _ = tx.send(AiMoveResult {
                mv: r.mv,
                elapsed: r.elapsed,
                sims: r.total_sims,
            });
        });
        self.ai_handle = Some(handle);
    }

    /// 当前难度在当前引擎下的档位序号（用于检测切换）。
    fn difficulty_index(&self) -> u8 {
        match self.ai_engine {
            AiEngine::Classic => match self.difficulty {
                Difficulty::Easy => 0,
                Difficulty::Medium => 1,
                Difficulty::Hard => 2,
                Difficulty::Master => 3,
            },
            AiEngine::Neural => match self.nn_difficulty {
                NnDifficulty::Low => 0,
                NnDifficulty::Medium => 1,
                NnDifficulty::High => 2,
            },
        }
    }

    /// 收取 AI 结果并应用。
    fn poll_ai(&mut self) {
        let Some(rx) = &self.ai_rx else {
            return;
        };
        if let Ok(result) = rx.try_recv() {
            self.ai_busy = false;
            // 结果已收到 → 释放线程句柄（线程已自然结束）
            if let Some(h) = self.ai_handle.take() {
                let _ = h.join();
            }
            self.apply_move(result.mv, Some(result));
        }
    }

    /// 应用一手棋（人类或 AI），并维护界面状态。
    fn apply_move(&mut self, mv: Move, ai: Option<AiMoveResult>) {
        if let Move::Play(p) = mv {
            if !self.game.play(mv) {
                self.message = format!("非法着法：{}", p.sgf());
                return;
            }
            self.last_played = Some(p);
            let n = self.game.move_count();
            self.move_numbers.insert(p, n);
            self.message = match ai {
                Some(r) => format!(
                    "电脑落子 {}（{:.2}s / {} 模拟）",
                    p.sgf(),
                    r.elapsed.as_secs_f64(),
                    r.sims
                ),
                None => format!("落子 {}", p.sgf()),
            };
        } else if let Some(r) = ai {
            match mv {
                Move::Pass => {
                    self.game.play(Move::Pass);
                    self.message = format!("电脑虚着（{:.2}s）", r.elapsed.as_secs_f64());
                }
                Move::Resign => {
                    self.game.play(Move::Resign);
                    self.message = "电脑认输".to_string();
                }
                _ => {}
            }
        }
        self.hint_cache = None;
    }

    /// 人类点击落子（复盘状态下先截断到游标处，再回到实时对弈）。
    fn human_play(&mut self, p: Point) {
        if let Some(idx) = self.replay_index {
            let idx = idx.min(self.game.move_count());
            while self.game.move_count() > idx {
                self.game.undo();
            }
            self.replay_index = None;
            self.hint_cache = None;
        }
        if self.game.status() == GameStatus::Scoring {
            if self.marking_dead {
                if self.game.mark_dead(p.x, p.y) {
                    self.message = format!("{} 死子标记已切换", p.sgf());
                }
            }
            return;
        }
        self.apply_move(Move::Play(p), None);
    }

    /// 计算合法点提示（基于当前显示局面，带缓存）。
    fn legal_hints(&mut self) -> Vec<Point> {
        let view = self.view_game();
        let board = view.board();
        let h = board.hash();
        if let Some((cached_h, cached)) = &self.hint_cache {
            if *cached_h == h {
                return cached.clone();
            }
        }
        let legal = board.legal_moves();
        self.hint_cache = Some((h, legal.clone()));
        legal
    }

    /// 悔棋：人机模式下撤回 AI + 人类两步，其余撤回一步。
    fn undo_move(&mut self) {
        if self.ai_busy {
            self.message = "AI 思考中，暂时无法悔棋".to_string();
            return;
        }
        let steps = match self.mode {
            GuiMode::HumanVsAi(_) => 2,
            _ => 1,
        };
        let mut undone = 0;
        for _ in 0..steps {
            if self.game.undo().is_some() {
                undone += 1;
            } else {
                break;
            }
        }
        if undone > 0 {
            self.last_played = self
                .game
                .last_move()
                .and_then(|(m, _)| if let Move::Play(p) = m { Some(p) } else { None });
            self.marking_dead = false;
            self.message = format!("已悔棋 {} 步", undone);
            self.hint_cache = None;
        } else {
            self.message = "没有可悔的棋".to_string();
        }
    }

    /// 判断当前局面胜负（实时计分，不改变对局状态）。
    /// 死子 = 手动标记 ∪ 自动判死；结果显示在状态栏与信息面板。
    fn judge_position(&mut self) {
        let mut dead = self.game.dead().clone();
        dead.extend(self.game.board().auto_dead_points());
        let s = compute_score(self.game.board(), &dead);
        let msg = match s.winner {
            Some(Color::Black) => format!("黑胜 {:.1} 子", s.margin),
            Some(Color::White) => format!("白胜 {:.1} 子", s.margin),
            _ => "和棋".to_string(),
        };
        self.message = format!(
            "局面判断：黑 {} 子（{}+{}）vs 白 {} 子（{}+{}）→ {msg}",
            s.black_total,
            s.black_stones,
            s.black_territory,
            s.white_total,
            s.white_stones,
            s.white_territory
        );
        self.judged_score = Some(s);
        // 弹出确认对话框（选择“确认结束”或“继续对弈”）
        self.confirm_dialog = Some(s);
    }

    /// 触发终局胜利/失败特效。
    fn trigger_effect(&mut self, winner: Option<Color>) {
        let human = match self.mode {
            GuiMode::HumanVsAi(c) => Some(c),
            _ => None,
        };
        self.effect = Some(VictoryEffect {
            winner,
            human_color: human,
            start: Instant::now(),
            duration: 3.5,
        });
    }

    /// 渲染“判断胜负”确认对话框（全屏遮罩 + 居中窗口）。
    fn render_confirm_dialog(&mut self, ctx: &egui::Context) {
        let Some(score) = self.confirm_dialog else { return; };
        let screen = ctx.screen_rect();
        // 遮罩层：捕获点击，阻止棋盘交互
        egui::Area::new(egui::Id::new("judge_dim"))
            .order(egui::Order::Middle)
            .show(ctx, |ui| {
                ui.allocate_rect(screen, egui::Sense::click());
                ui.painter()
                    .rect_filled(screen, 0.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 150));
            });
        // 对话框
        egui::Window::new("判断结果")
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.set_min_width(300.0);
                ui.label(
                    egui::RichText::new("当前局面胜负判断")
                        .size(15.0)
                        .strong()
                        .color(egui::Color32::from_rgb(217, 160, 91)),
                );
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(format!(
                        "黑方：{} 子 + {} 空 = {}",
                        score.black_stones, score.black_territory, score.black_total
                    ))
                    .size(13.0),
                );
                ui.label(
                    egui::RichText::new(format!(
                        "白方：{} 子 + {} 空 = {}",
                        score.white_stones, score.white_territory, score.white_total
                    ))
                    .size(13.0),
                );
                ui.add_space(6.0);
                let result = match score.winner {
                    Some(Color::Black) => format!("黑胜 {:.1} 子", score.margin),
                    Some(Color::White) => format!("白胜 {:.1} 子", score.margin),
                    _ => "和棋".to_string(),
                };
                ui.label(
                    egui::RichText::new(result)
                        .size(20.0)
                        .strong()
                        .color(egui::Color32::from_rgb(235, 176, 106)),
                );
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui.button("确认结束").clicked() {
                        // 将判定所用的死子写入对局并确认终局
                        let mut dead = self.game.dead().clone();
                        dead.extend(self.game.board().auto_dead_points());
                        self.game.set_dead(dead);
                        let s = self.game.confirm_score();
                        self.message = match s.winner {
                            Some(Color::Black) => format!("终局：黑胜 {:.1} 子", s.margin),
                            Some(Color::White) => format!("终局：白胜 {:.1} 子", s.margin),
                            _ => "终局：和棋".to_string(),
                        };
                        self.trigger_effect(s.winner);
                        self.confirm_dialog = None;
                        self.marking_dead = false;
                        self.hint_cache = None;
                    }
                    if ui.button("继续对弈").clicked() {
                        self.confirm_dialog = None;
                    }
                });
            });
    }

    /// 渲染终局胜利/失败特效（粒子彩带 + 发光大标题，约 3.5 秒）。
    fn render_effect(&mut self, ctx: &egui::Context) {
        let (winner, human_color, start, duration) = match &self.effect {
            Some(e) => (e.winner, e.human_color, e.start, e.duration),
            None => return,
        };
        let elapsed = start.elapsed().as_secs_f32();
        if elapsed >= duration {
            self.effect = None;
            return;
        }
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("victory_effect"),
        ));
        let rect = ctx.screen_rect();
        let center = rect.center();
        // 暖色闪烁背景
        let flash = ((elapsed * 6.0).sin().abs() * 30.0) as u8;
        painter.rect_filled(
            rect,
            0.0,
            egui::Color32::from_rgba_unmultiplied(255, 190, 80, flash),
        );
        // 粒子彩带（确定性伪随机）
        let mut state: u64 = 0xEFFE_2026;
        let colors = [
            (255, 80, 80),
            (255, 180, 60),
            (255, 240, 80),
            (120, 255, 120),
            (90, 180, 255),
            (230, 120, 255),
        ];
        for i in 0..70u32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let sx = ((state >> 33) as f32) / 2147483648.0;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let sy = ((state >> 33) as f32) / 2147483648.0;
            let x0 = rect.left() + sx * rect.width();
            let y0 = rect.bottom() - sy * rect.height() * 0.35;
            let dx = (sx - 0.5) * 2.0 * 240.0 * elapsed;
            let dy = -160.0 * elapsed - 280.0 * elapsed * elapsed;
            let alpha = ((1.0 - elapsed / duration) * 255.0) as u8;
            let (r, g, b) = colors[(i % 6) as usize];
            painter.circle_filled(
                egui::pos2(x0 + dx, y0 + dy),
                3.5 + 3.5 * sy,
                egui::Color32::from_rgba_unmultiplied(r, g, b, alpha),
            );
        }
        // 大标题（带投影发光）
        let (text, color) = match (human_color, winner) {
            (Some(h), Some(w)) if w == h => ("胜利！", egui::Color32::from_rgb(255, 214, 90)),
            (Some(_), Some(_)) => ("失败…", egui::Color32::from_rgb(255, 120, 110)),
            (_, Some(Color::Black)) => ("黑胜！", egui::Color32::from_rgb(255, 214, 90)),
            (_, Some(Color::White)) => ("白胜！", egui::Color32::from_rgb(240, 240, 240)),
            _ => ("和棋", egui::Color32::from_rgb(200, 200, 200)),
        };
        let font_size = (rect.height() * 0.09).clamp(36.0, 90.0);
        let font = egui::FontId::proportional(font_size);
        painter.text(
            center + egui::vec2(4.0, 4.0),
            egui::Align2::CENTER_CENTER,
            text,
            font.clone(),
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 150),
        );
        painter.text(center, egui::Align2::CENTER_CENTER, text, font, color);
        ctx.request_repaint();
    }

    /// 当前显示的对局（复盘游标生效时截断到游标处；否则为实时对局）。
    fn view_game(&self) -> Game {
        let mut g = self.game.clone();
        if let Some(idx) = self.replay_index {
            let idx = idx.min(g.move_count());
            while g.move_count() > idx {
                g.undo();
            }
        }
        g
    }

    /// 按视图对局重建手数映射（供棋子上编号显示）。
    fn rebuild_move_numbers(&mut self, view: &Game) {
        self.move_numbers.clear();
        let mut n = 0usize;
        for (m, _) in view.history() {
            if let Move::Play(p) = m {
                n += 1;
                self.move_numbers.insert(*p, n);
            }
        }
    }

    /// 设置复盘游标（None = 回到实时对局）。
    fn set_replay(&mut self, idx: Option<usize>) {
        self.replay_index = idx;
        let view = self.view_game();
        self.rebuild_move_numbers(&view);
        self.last_played = view
            .last_move()
            .and_then(|(m, _)| if let Move::Play(p) = m { Some(p) } else { None });
        self.hint_cache = None;
    }

    /// 保存当前对局为 SGF 棋谱（games/ 目录）。
    fn save_sgf(&mut self) {
        let _ = std::fs::create_dir_all(&self.games_dir);
        let mut name = self.save_name.trim().to_string();
        if name.is_empty() {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            name = format!("对局_{}", ts);
        }
        let name: String = name
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let path = format!("{}/{}.sgf", self.games_dir, name);
        let sgf = crate::sgf::to_sgf(&self.game);
        match std::fs::write(&path, sgf) {
            Ok(_) => {
                self.message = format!("棋谱已保存：{}", path);
                self.save_name = name;
            }
            Err(e) => self.message = format!("保存失败：{e}"),
        }
        self.refresh_sgf_files();
    }

    /// 刷新 games/ 目录中的棋谱文件列表。
    fn refresh_sgf_files(&mut self) {
        let mut files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.games_dir) {
            for e in entries.flatten() {
                let path = e.path();
                if path.extension().map_or(false, |x| x == "sgf") {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        files.push(name.to_string());
                    }
                }
            }
        }
        files.sort();
        self.sgf_files = files;
        if let Some(sel) = &self.selected_sgf {
            if !self.sgf_files.contains(sel) {
                self.selected_sgf = None;
            }
        }
    }

    /// 加载选中的棋谱进行复盘（重建对局并进入复盘模式）。
    fn load_selected_sgf(&mut self) {
        let Some(sel) = self.selected_sgf.clone() else {
            self.message = "请先在列表中选择棋谱".to_string();
            return;
        };
        let path = format!("{}/{}", self.games_dir, sel);
        match std::fs::read_to_string(&path) {
            Ok(text) => match crate::sgf::game_from_sgf(&text) {
                Ok(game) => {
                    self.game = game;
                    self.judged_score = None;
                    self.marking_dead = false;
                    self.ai_busy = false;
                    let n = self.game.move_count();
                    self.set_replay(Some(n));
                    self.message = format!("已加载棋谱：{}（{} 手）", sel, n);
                }
                Err(e) => self.message = format!("棋谱解析失败：{e}"),
            },
            Err(e) => self.message = format!("读取失败：{e}"),
        }
    }
}

impl Drop for GoApp {
    fn drop(&mut self) {
        // 窗口关闭时：请求 AI 取消 + 等待 AI 线程退出（防止进程挂起）。
        // 设置全局取消标志 → AI 线程每 64 次迭代检查一次，会在数毫秒内退出。
        request_ai_cancel();
        if let Some(h) = self.ai_handle.take() {
            let deadline = Instant::now() + std::time::Duration::from_secs(2);
            loop {
                if h.is_finished() {
                    let _ = h.join();
                    break;
                }
                if Instant::now() >= deadline {
                    // 超时未退出：线程仍会在进程退出时被 OS 杀死，不阻塞析构
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        // 清除全局取消标志，供下次启动使用
        clear_ai_cancel();
    }
}

impl eframe::App for GoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 收取 AI 结果
        self.poll_ai();

        // ---------- 顶栏：标题 ----------
        egui::TopBottomPanel::top("title_bar")
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(38, 32, 26))
                    .inner_margin(egui::Margin::symmetric(14.0, 10.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("GobanGo 围棋")
                            .size(22.0)
                            .strong()
                            .color(egui::Color32::from_rgb(217, 160, 91)),
                    );
                    ui.separator();
                    ui.label(
                        egui::RichText::new("19 路 · 中国规则 · 三难度人机")
                            .size(12.0)
                            .color(egui::Color32::from_rgb(150, 140, 125)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("难度：{}", self.difficulty.label()))
                                .size(13.0)
                                .color(egui::Color32::from_rgb(217, 160, 91)),
                        );
                    });
                });
            });

        // ---------- 顶栏：控制区 ----------
        egui::TopBottomPanel::top("controls")
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(46, 39, 32))
                    .inner_margin(egui::Margin::symmetric(14.0, 8.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("模式").color(egui::Color32::from_rgb(180, 170, 155)),
                    );
                    for (label, mode) in [
                        ("双人", GuiMode::HumanVsHuman),
                        ("人机·执黑", GuiMode::HumanVsAi(Color::Black)),
                        ("人机·执白", GuiMode::HumanVsAi(Color::White)),
                        ("AI观战", GuiMode::AiVsAi),
                    ] {
                        if ui
                            .selectable_label(self.mode == mode, label)
                            .on_hover_text(match mode {
                                GuiMode::HumanVsAi(_) => "人机对弈（可选手色）",
                                GuiMode::AiVsAi => "观战：两个 AI 自动对弈",
                                _ => "两名玩家轮流落子",
                            })
                            .clicked()
                        {
                            self.mode = mode;
                            self.new_game();
                        }
                    }
                    ui.separator();
                    ui.label(
                        egui::RichText::new("AI 引擎").color(egui::Color32::from_rgb(180, 170, 155)),
                    );
                    ui.horizontal(|ui| {
                        for e in [AiEngine::Classic, AiEngine::Neural] {
                            if ui
                                .selectable_label(self.ai_engine == e, e.label())
                                .on_hover_text(e.description())
                                .clicked()
                            {
                                self.ai_engine = e;
                            }
                        }
                    });
                    ui.label(
                        egui::RichText::new("难度").color(egui::Color32::from_rgb(180, 170, 155)),
                    );
                    match self.ai_engine {
                        AiEngine::Classic => {
                            for d in [
                                Difficulty::Easy,
                                Difficulty::Medium,
                                Difficulty::Hard,
                                Difficulty::Master,
                            ] {
                                if ui
                                    .selectable_label(self.difficulty == d, d.label())
                                    .on_hover_text(d.description())
                                    .clicked()
                                {
                                    self.difficulty = d;
                                }
                            }
                        }
                        AiEngine::Neural => {
                            for d in NnDifficulty::all() {
                                if ui
                                    .selectable_label(self.nn_difficulty == d, d.label())
                                    .on_hover_text(d.description())
                                    .clicked()
                                {
                                    self.nn_difficulty = d;
                                }
                            }
                        }
                    }
                    ui.separator();
                    if ui.button("新游戏").clicked() {
                        self.new_game();
                    }
                    if ui.button("悔棋").clicked() {
                        self.undo_move();
                    }
                    let passable = self.game.status() == GameStatus::Ongoing && !self.ai_busy;
                    if ui.add_enabled(passable, egui::Button::new("虚着")).clicked() {
                        self.apply_move(Move::Pass, None);
                        self.message = "虚着（Pass）".to_string();
                    }
                    if ui.add_enabled(passable, egui::Button::new("认输")).clicked() {
                        let c = self.game.current_player();
                        self.game.play(Move::Resign);
                        self.message = format!("{}认输，{}胜", c.label(), c.opponent().label());
                        self.trigger_effect(Some(c.opponent()));
                        self.hint_cache = None;
                    }
                    ui.separator();
                    if ui
                        .button("判断胜负")
                        .on_hover_text("按当前局面实时计分（自动判死 + 手动标记），不改变对局状态")
                        .clicked()
                    {
                        self.judge_position();
                    }
                });
                // 第二行：棋盘尺寸
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("棋盘尺寸")
                            .color(egui::Color32::from_rgb(180, 170, 155)),
                    );
                    for (label, px) in [
                        ("600×600", 600.0f32),
                        ("800×800", 800.0),
                        ("1080×1080", 1080.0),
                        ("2046×2046", 2046.0),
                        ("自适应", 0.0),
                    ] {
                        let selected = (self.board_px - px).abs() < 0.5;
                        if ui
                            .selectable_label(selected, label)
                            .on_hover_text(match px {
                                0.0 => "棋盘随窗口自动缩放".to_string(),
                                _ => format!("棋盘固定 {px:.0}×{px:.0} 像素"),
                            })
                            .clicked()
                        {
                            self.board_px = px;
                            if px > 0.0 {
                                // 窗口宽度 = 棋盘 + 右侧面板(280) + 边距
                                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(
                                    egui::vec2(px + 310.0, px + 180.0),
                                ));
                            } else {
                                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(
                                    egui::vec2(1000.0, 860.0),
                                ));
                            }
                        }
                    }
                    ui.separator();
                    // GPU 计算勾选项：勾选后由 GPU 批量并行走子（无 GPU 自动回退 CPU）
                    let mut gpu_checked = self.gpu_engine.is_some();
                    if ui
                        .checkbox(&mut gpu_checked, "GPU 计算")
                        .on_hover_text("勾选后走子阶段由 GPU 并行计算（无 GPU/驱动时自动回退 CPU）")
                        .changed()
                    {
                        if gpu_checked {
                            match GpuEngine::try_new() {
                                Ok(engine) => {
                                    self.gpu_engine = Some(Arc::new(Mutex::new(engine)));
                                    self.message =
                                        "GPU 计算已启用（走子由 GPU 批量并行计算）".to_string();
                                }
                                Err(e) => {
                                    self.gpu_engine = None;
                                    self.message = format!("GPU 不可用，已回退 CPU：{e}");
                                }
                            }
                        } else {
                            self.gpu_engine = None;
                            self.message = "已切换回 CPU 计算".to_string();
                        }
                    }
                });
                // 第三行：复盘 + 棋谱
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("复盘")
                            .color(egui::Color32::from_rgb(180, 170, 155)),
                    );
                    let len = self.game.move_count();
                    let mut idx = self.replay_index.unwrap_or(len);
                    let replaying = self.replay_index.is_some();
                    if ui
                        .button("⏮")
                        .on_hover_text("回到第一手")
                        .clicked()
                    {
                        self.set_replay(Some(0));
                    }
                    if ui
                        .button("◀")
                        .on_hover_text("上一手")
                        .clicked()
                    {
                        self.set_replay(Some(idx.saturating_sub(1)));
                    }
                    let slider_text = format!("{idx}/{len}");
                    let slider = ui.add(
                        egui::Slider::new(&mut idx, 0..=len)
                            .show_value(true)
                            .text(slider_text),
                    );
                    if slider.changed() {
                        self.set_replay(Some(idx));
                    }
                    if ui
                        .button("▶")
                        .on_hover_text("下一手")
                        .clicked()
                    {
                        self.set_replay(Some((idx + 1).min(len)));
                    }
                    if ui
                        .button("⏭")
                        .on_hover_text("跳到最后")
                        .clicked()
                    {
                        self.set_replay(Some(len));
                    }
                    if replaying {
                        if ui.button("返回对弈").clicked() {
                            self.set_replay(None);
                        }
                    }
                    ui.separator();
                    // 棋谱保存
                    ui.add(
                        egui::TextEdit::singleline(&mut self.save_name)
                            .hint_text("棋谱文件名")
                            .desired_width(110.0),
                    );
                    if ui.button("保存棋谱").clicked() {
                        self.save_sgf();
                    }
                    ui.separator();
                    // 棋谱加载（从 games/ 目录选择）
                    let sel_text = self
                        .selected_sgf
                        .clone()
                        .unwrap_or_else(|| "选择棋谱…".to_string());
                    egui::ComboBox::from_id_source("sgf_files")
                        .selected_text(sel_text)
                        .width(160.0)
                        .show_ui(ui, |ui| {
                            if self.sgf_files.is_empty() {
                                ui.label("（暂无棋谱，先保存一局）");
                            }
                            for f in self.sgf_files.clone() {
                                ui.selectable_value(&mut self.selected_sgf, Some(f.clone()), f);
                            }
                        });
                    if ui.button("加载复盘").clicked() {
                        self.load_selected_sgf();
                    }
                    if ui.button("刷新").clicked() {
                        self.refresh_sgf_files();
                    }
                });
            });

        // ---------- 底栏：状态与消息 ----------
        egui::TopBottomPanel::bottom("status")
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(38, 32, 26))
                    .inner_margin(egui::Margin::symmetric(14.0, 8.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let turn = self.game.current_player();
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(format!("第 {} 手", self.game.move_count() + 1))
                                .size(14.0),
                        ),
                    );
                    ui.separator();
                    let status_text = match self.game.status() {
                        GameStatus::Ongoing => format!(
                            "{}行棋 · 黑提 {} · 白提 {}",
                            turn.label(),
                            self.game.board().black_captures(),
                            self.game.board().white_captures()
                        ),
                        GameStatus::Scoring => "计分阶段：可标记死子".to_string(),
                        GameStatus::Finished => match self.game.winner() {
                            Some(w) => format!("对局结束：{}胜", w.label()),
                            None => "对局结束：和棋".to_string(),
                        },
                    };
                    ui.label(
                        egui::RichText::new(status_text)
                            .size(14.0)
                            .color(egui::Color32::from_rgb(210, 200, 185)),
                    );
                });
                ui.horizontal(|ui| {
                    ui.colored_label(
                        egui::Color32::from_rgb(235, 176, 106),
                        &self.message,
                    );
                    if self.ai_busy {
                        ui.spinner();
                    }
                });
            });

        // ---------- 中央：棋盘 + 计分 ----------
        // 左右双栏布局：左为棋盘（固定尺寸或自适应），右为信息面板（强制垂直堆叠 + 可滚动）。
        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(30, 26, 22))
                    .inner_margin(egui::Margin::same(8.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // 左：棋盘
                    self.board_area(ui, ctx);
                    ui.add_space(6.0);
                    // 右：信息面板 —— 强制垂直堆叠 + 整体可滚动
                    ui.vertical(|ui| {
                        egui::ScrollArea::vertical()
                            .id_source("right_panel_scroll")
                            .auto_shrink([false, false])
                            .max_width(320.0)
                            .show(ui, |ui| {
                                ui.set_min_width(280.0);
                                self.info_panel(ui);
                            });
                    });
                });
            });

        // 触发 AI 行棋
        if self.is_ai_turn() && !self.ai_busy {
            let paced = self.mode == GuiMode::AiVsAi
                && self.last_ai_time.elapsed() < std::time::Duration::from_millis(250);
            if !paced {
                self.last_ai_time = Instant::now();
                self.start_ai();
            }
        }
        if self.ai_busy || (self.mode == GuiMode::AiVsAi && self.is_ai_turn()) {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }

        // 判断胜负确认对话框（遮罩层）
        self.render_confirm_dialog(ctx);
        // 终局胜利/失败特效
        self.render_effect(ctx);
    }
}

impl GoApp {
    /// 绘制棋盘区域（含交互）。
    fn board_area(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        // 右侧面板固定 280px + 6px gap；自适应模式下棋盘只用剩余宽度
        let panel_avail = ui.available_size();
        let right_panel_w = 286.0f32;
        let board_avail_w = (panel_avail.x - right_panel_w).max(200.0);
        let board_avail_h = panel_avail.y;

        let (board_px, cell) = if self.board_px > 0.0 {
            let cell = (self.board_px - 44.0) / (BOARD_SIZE as f32 - 1.0);
            (self.board_px, cell)
        } else {
            let cell = (((board_avail_w - 40.0) / (BOARD_SIZE as f32 - 1.0))
                .min((board_avail_h - 40.0) / (BOARD_SIZE as f32 - 1.0)))
            .min(38.0)
            .max(16.0);
            (cell * (BOARD_SIZE as f32 - 1.0) + 44.0, cell)
        };
        let (response, painter) =
            ui.allocate_painter(egui::vec2(board_px, board_px), egui::Sense::click());
        let rect = response.rect;
        let margin = 22.0;
        let origin = egui::pos2(rect.left() + margin, rect.top() + margin);

        // 使用复盘视图渲染（游标生效时显示截断局面）
        let view = self.view_game();
        self.rebuild_move_numbers(&view);

        // 木纹渐变背景
        paint_wood_background(&painter, rect);

        // 坐标标签
        let coord_color = egui::Color32::from_rgb(90, 62, 34);
        for i in 0..BOARD_SIZE {
            let letter = if i < 8 { (b'A' + i as u8) as char } else { (b'A' + i as u8 + 1) as char };
            let x = origin.x + i as f32 * cell;
            let y = origin.y + i as f32 * cell;
            painter.text(
                egui::pos2(x, rect.top() + 8.0),
                egui::Align2::CENTER_CENTER,
                letter.to_string(),
                egui::FontId::proportional(11.0),
                coord_color,
            );
            painter.text(
                egui::pos2(rect.left() + 8.0, y),
                egui::Align2::CENTER_CENTER,
                (BOARD_SIZE - i).to_string(),
                egui::FontId::proportional(11.0),
                coord_color,
            );
        }

        // 网格线
        let line_color = egui::Color32::from_rgb(70, 46, 22);
        for i in 0..BOARD_SIZE {
            let a = egui::pos2(origin.x, origin.y + i as f32 * cell);
            let b = egui::pos2(origin.x + (BOARD_SIZE as f32 - 1.0) * cell, origin.y + i as f32 * cell);
            painter.line_segment([a, b], egui::Stroke::new(1.0, line_color));
            let a = egui::pos2(origin.x + i as f32 * cell, origin.y);
            let b = egui::pos2(origin.x + i as f32 * cell, origin.y + (BOARD_SIZE as f32 - 1.0) * cell);
            painter.line_segment([a, b], egui::Stroke::new(1.0, line_color));
        }

        // 星位
        for &(x, y) in &STAR_POINTS {
            let c = egui::pos2(origin.x + x as f32 * cell, origin.y + y as f32 * cell);
            painter.circle_filled(c, cell * 0.11, line_color);
        }

        // 最近一手：呼吸式光晕（绘制在棋子下层，形成发光效果）
        let stone_r = cell * 0.46;
        let glow_t = ctx.input(|i| i.time);
        let pulse = 0.55f32 + 0.45 * ((glow_t as f32) * 4.0).sin();
        if let Some(p) = self.last_played {
            let center = egui::pos2(origin.x + p.x as f32 * cell, origin.y + p.y as f32 * cell);
            for (k, base_a) in [(2.1, 22.0f32), (1.7, 42.0), (1.3, 80.0)] {
                painter.circle_filled(
                    center,
                    stone_r * k,
                    egui::Color32::from_rgba_unmultiplied(
                        255,
                        120,
                        30,
                        (base_a * pulse) as u8,
                    ),
                );
            }
        }

        // 棋子 + 编号 + 死子标记
        let board = view.board();
        let dead = view.dead().clone();
        for y in 0..BOARD_SIZE {
            for x in 0..BOARD_SIZE {
                let c = board.at(x, y);
                if c == Color::Empty {
                    continue;
                }
                let center = egui::pos2(origin.x + x as f32 * cell, origin.y + y as f32 * cell);
                paint_stone(&painter, center, stone_r, c);
                if let Some(n) = self.move_numbers.get(&Point::new(x, y)) {
                    let text_color = match c {
                        Color::Black => egui::Color32::from_rgb(210, 210, 210),
                        _ => egui::Color32::from_rgb(70, 60, 50),
                    };
                    painter.text(
                        center,
                        egui::Align2::CENTER_CENTER,
                        n.to_string(),
                        egui::FontId::proportional((stone_r * 0.95).max(7.0)),
                        text_color,
                    );
                }
                if dead.contains(&(x, y)) {
                    let d = stone_r * 0.7;
                    painter.line_segment(
                        [egui::pos2(center.x - d, center.y - d), egui::pos2(center.x + d, center.y + d)],
                        egui::Stroke::new(2.2, egui::Color32::from_rgb(220, 60, 50)),
                    );
                    painter.line_segment(
                        [egui::pos2(center.x - d, center.y + d), egui::pos2(center.x + d, center.y - d)],
                        egui::Stroke::new(2.2, egui::Color32::from_rgb(220, 60, 50)),
                    );
                }
            }
        }

        // 最近一手：中心亮点（棋子上层）
        if let Some(p) = self.last_played {
            let center = egui::pos2(origin.x + p.x as f32 * cell, origin.y + p.y as f32 * cell);
            painter.circle_filled(center, 3.4, egui::Color32::from_rgb(255, 214, 120));
            painter.circle_stroke(
                center,
                3.4,
                egui::Stroke::new(1.2, egui::Color32::from_rgb(255, 140, 40)),
            );
        }

        // 悬停预览 + 合法点提示（人类回合且非计分标记状态）
        if let Some(pos) = response.hover_pos() {
            let col = ((pos.x - origin.x) / cell).round() as i32;
            let row = ((pos.y - origin.y) / cell).round() as i32;
            if col >= 0 && row >= 0 && col < BOARD_SIZE as i32 && row < BOARD_SIZE as i32 {
                self.hover = Some(Point::new(col as usize, row as usize));
            } else {
                self.hover = None;
            }
        }
        let human_turn = !self.is_ai_turn() && self.game.status() == GameStatus::Ongoing;
        if human_turn {
            for p in self.legal_hints() {
                let c = egui::pos2(origin.x + p.x as f32 * cell, origin.y + p.y as f32 * cell);
                painter.circle_filled(
                    c,
                    cell * 0.09,
                    egui::Color32::from_rgba_unmultiplied(30, 24, 18, 70),
                );
            }
            if let Some(h) = self.hover {
                if view.board().is_legal(h.x, h.y) {
                    let c = egui::pos2(origin.x + h.x as f32 * cell, origin.y + h.y as f32 * cell);
                    let t = self.game.current_player();
                    paint_stone_translucent(&painter, c, stone_r, t);
                }
            }
        }

        // 点击处理
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let col = ((pos.x - origin.x) / cell).round() as i32;
                let row = ((pos.y - origin.y) / cell).round() as i32;
                if col >= 0 && row >= 0 && col < BOARD_SIZE as i32 && row < BOARD_SIZE as i32 {
                    let p = Point::new(col as usize, row as usize);
                    if self.game.status() == GameStatus::Scoring && !self.marking_dead {
                        // 计分阶段默认仍可点按：弹出提示
                        self.message = "请先点击“标记死子”再点选棋子".to_string();
                    } else {
                        self.human_play(p);
                    }
                }
            }
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(80));
    }

    /// 右侧信息面板：按上下顺序依次为
    /// ① 当前/下一步信息（第 N+1 手 · 轮到谁 · 位置待定 / 最近一手位置）
    /// ② 落子记录 log（完整落子序列，可滚动）
    /// ③ 对弈信息（模式 / 难度 / 行数 / 提子 / 贴目 / 计算方式）
    /// 其余（复盘状态、判断、计分、终局）按需追加。
    fn info_panel(&mut self, ui: &mut egui::Ui) {
        ui.set_min_width(230.0);
        ui.add_space(4.0);

        // 卡片 0（最顶）：当前/下一步信息 —— 第几步 / 谁在下 / 什么位置
        let turn = self.game.current_player();
        let last_move_info = self
            .game
            .history()
            .last()
            .map(|(m, c)| (m.clone(), *c))
            .unwrap_or((Move::Pass, Color::Black));
        egui::Frame::none()
            .fill(egui::Color32::from_rgb(60, 52, 44))
            .rounding(egui::Rounding::same(8.0))
            .inner_margin(egui::Margin::same(12.0))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("当前 / 下一步")
                            .size(13.0)
                            .strong()
                            .color(egui::Color32::from_rgb(217, 160, 91)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let (dot_p, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                        let dot = match turn {
                            Color::Black => egui::Color32::from_rgb(30, 30, 34),
                            _ => egui::Color32::from_rgb(240, 240, 240),
                        };
                        ui.painter().circle_filled(dot_p.center(), 6.0, dot);
                        ui.painter().circle_stroke(
                            dot_p.center(),
                            6.0,
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(160, 150, 135)),
                        );
                    });
                });
                ui.add_space(4.0);
                // 第几步 · 谁在下
                let n = self.game.move_count() + 1;
                let status = match self.game.status() {
                    GameStatus::Ongoing => format!("轮到 {} 行棋", turn.label()),
                    GameStatus::Scoring => "计分阶段".to_string(),
                    GameStatus::Finished => "对局已结束".to_string(),
                };
                ui.label(
                    egui::RichText::new(format!("第 {n} 手　·　{status}"))
                        .size(14.0)
                        .strong()
                        .color(egui::Color32::from_rgb(235, 176, 106)),
                );
                ui.add_space(2.0);
                // 最近一手位置 / 下一步位置待定
                let (label_pos, color_pos) = match &last_move_info.0 {
                    Move::Play(pp) => (format!("最近一手：{} （{}）", pp.sgf(), last_move_info.1.label()), egui::Color32::from_rgb(200, 192, 178)),
                    Move::Pass => ("最近：虚着".to_string(), egui::Color32::from_rgb(180, 170, 155)),
                    Move::Resign => ("最近：认输".to_string(), egui::Color32::from_rgb(180, 170, 155)),
                };
                ui.label(egui::RichText::new(label_pos).size(12.0).color(color_pos));
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("下一步位置：")
                            .size(11.0)
                            .color(egui::Color32::from_rgb(160, 150, 135)),
                    );
                    let next_pos_text = match self.game.status() {
                        GameStatus::Ongoing => {
                            if self.ai_busy {
                                "AI 思考中…".to_string()
                            } else if let GuiMode::HumanVsAi(human) = self.mode {
                                if turn != human {
                                    "（电脑将落子）".to_string()
                                } else {
                                    "（点击棋盘落子）".to_string()
                                }
                            } else {
                                "（点击棋盘落子）".to_string()
                            }
                        }
                        GameStatus::Scoring => "—".to_string(),
                        GameStatus::Finished => "—".to_string(),
                    };
                    ui.label(
                        egui::RichText::new(next_pos_text)
                            .size(11.0)
                            .color(egui::Color32::from_rgb(140, 130, 115)),
                    );
                });
            });
        ui.add_space(6.0);

        // 卡片 1：落子记录 log（完整落子序列，可滚动，自动跟随最新一手）
        egui::Frame::none()
            .fill(egui::Color32::from_rgb(46, 39, 32))
            .rounding(egui::Rounding::same(8.0))
            .inner_margin(egui::Margin::same(12.0))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("落子记录")
                            .size(14.0)
                            .strong()
                            .color(egui::Color32::from_rgb(217, 160, 91)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("共 {} 手", self.game.move_count()))
                                .size(11.0)
                                .color(egui::Color32::from_rgb(150, 140, 125)),
                        );
                    });
                });
                ui.add_space(4.0);
                let hist = self.game.history();
                let n = self.game.move_count();
                egui::ScrollArea::vertical()
                    .id_source("move_log_scroll")
                    .max_height(150.0)
                    .stick_to_bottom(true)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        if hist.is_empty() {
                            ui.label(
                                egui::RichText::new("（暂无着法）")
                                    .size(11.0)
                                    .color(egui::Color32::from_rgb(140, 130, 115)),
                            );
                        }
                        for (i, (mv, color)) in hist.iter().enumerate() {
                            let is_last = i + 1 == n;
                            let sgf = match mv {
                                Move::Play(pp) => pp.sgf(),
                                Move::Pass => "虚着".to_string(),
                                Move::Resign => "认输".to_string(),
                            };
                            ui.horizontal(|ui| {
                                let (p, _) =
                                    ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                                let dot = match color {
                                    Color::Black => egui::Color32::from_rgb(30, 30, 34),
                                    _ => egui::Color32::from_rgb(240, 240, 240),
                                };
                                ui.painter().circle_filled(p.center(), 5.0, dot);
                                ui.painter().circle_stroke(
                                    p.center(),
                                    5.0,
                                    egui::Stroke::new(1.0, egui::Color32::from_rgb(150, 140, 125)),
                                );
                                let text = if is_last {
                                    egui::RichText::new(format!("{i}　{sgf}"))
                                        .size(12.0)
                                        .strong()
                                        .color(egui::Color32::from_rgb(235, 176, 106))
                                } else {
                                    egui::RichText::new(format!("{i}　{sgf}"))
                                        .size(12.0)
                                        .color(egui::Color32::from_rgb(200, 192, 178))
                                };
                                ui.label(text);
                            });
                        }
                    });
            });
        ui.add_space(8.0);

        // 卡片 1：对弈信息（网格布局，紧凑饱满）
        egui::Frame::none()
            .fill(egui::Color32::from_rgb(46, 39, 32))
            .rounding(egui::Rounding::same(8.0))
            .inner_margin(egui::Margin::same(12.0))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("对弈信息")
                        .size(14.0)
                        .strong()
                        .color(egui::Color32::from_rgb(217, 160, 91)),
                );
                ui.add_space(6.0);
                let label_color = egui::Color32::from_rgb(170, 160, 145);
                let value_color = egui::Color32::from_rgb(225, 218, 205);
                egui::Grid::new("info_grid")
                    .num_columns(2)
                    .spacing([14.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("模式").color(label_color));
                        ui.label(egui::RichText::new(self.mode.label()).color(value_color));
                        ui.end_row();
                        ui.label(egui::RichText::new("难度").color(label_color));
                        ui.label(
                            egui::RichText::new(format!(
                                "{} · 单步≤{}s",
                                self.difficulty.label(),
                                match self.difficulty {
                                    Difficulty::Easy => "0.5",
                                    Difficulty::Medium => "2.0",
                                    Difficulty::Hard => "5.0",
                                    Difficulty::Master => "60",
                                }
                            ))
                            .color(value_color),
                        );
                        ui.end_row();
                        ui.label(egui::RichText::new("行棋方").color(label_color));
                        ui.horizontal(|ui| {
                            let turn = self.game.current_player();
                            let (p, _) = ui
                                .allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
                            let dot = match turn {
                                Color::Black => egui::Color32::from_rgb(30, 30, 34),
                                _ => egui::Color32::from_rgb(240, 240, 240),
                            };
                            ui.painter().circle_filled(p.center(), 8.0, dot);
                            ui.painter().circle_stroke(
                                p.center(),
                                8.0,
                                egui::Stroke::new(1.2, egui::Color32::from_rgb(160, 150, 135)),
                            );
                            let s = match self.game.status() {
                                GameStatus::Ongoing => format!("{}行棋", turn.label()),
                                GameStatus::Scoring => "计分阶段".to_string(),
                                GameStatus::Finished => "对局已结束".to_string(),
                            };
                            ui.label(egui::RichText::new(s).size(14.0).strong().color(value_color));
                        });
                        ui.end_row();
                        ui.label(egui::RichText::new("手数").color(label_color));
                        ui.label(
                            egui::RichText::new(self.game.move_count().to_string())
                                .color(value_color),
                        );
                        ui.end_row();
                        ui.label(egui::RichText::new("黑提 / 白提").color(label_color));
                        ui.label(
                            egui::RichText::new(format!(
                                "{} / {}",
                                self.game.board().black_captures(),
                                self.game.board().white_captures()
                            ))
                            .color(value_color),
                        );
                        ui.end_row();
                        ui.label(egui::RichText::new("贴目").color(label_color));
                        ui.label(
                            egui::RichText::new(format!("{:.2} 子", self.game.komi()))
                                .color(value_color),
                        );
                        ui.end_row();
                        ui.label(egui::RichText::new("计算").color(label_color));
                        ui.label(
                            egui::RichText::new(if self.gpu_engine.is_some() {
                                "GPU（并行走子）"
                            } else {
                                "CPU"
                            })
                            .color(value_color),
                        );
                        ui.end_row();
                    });
            });
        ui.add_space(8.0);

        // 复盘状态指示
        if let Some(idx) = self.replay_index {
            let len = self.game.move_count();
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(70, 54, 34))
                .rounding(egui::Rounding::same(6.0))
                .inner_margin(egui::Margin::same(8.0))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(format!("📋 复盘模式：第 {idx} / {len} 手"))
                            .size(13.0)
                            .strong()
                            .color(egui::Color32::from_rgb(235, 176, 106)),
                    );
                    ui.label(
                        egui::RichText::new("点击棋盘可从该局面续下（截断后续着法）")
                            .size(11.0)
                            .color(egui::Color32::from_rgb(180, 170, 155)),
                    );
                });
            ui.add_space(6.0);
        }

        // 判断胜负结果（临时展示，不改变对局状态）
        if let Some(s) = self.judged_score {
            ui.separator();
            ui.add_space(4.0);
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(52, 44, 36))
                .rounding(egui::Rounding::same(8.0))
                .inner_margin(egui::Margin::same(10.0))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("局面判断（实时）")
                            .size(13.0)
                            .strong()
                            .color(egui::Color32::from_rgb(235, 176, 106)),
                    );
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(format!(
                            "黑：{} 子 + {} 空 = {}",
                            s.black_stones, s.black_territory, s.black_total
                        ))
                        .size(12.5),
                    );
                    ui.label(
                        egui::RichText::new(format!(
                            "白：{} 子 + {} 空 = {}",
                            s.white_stones, s.white_territory, s.white_total
                        ))
                        .size(12.5),
                    );
                    let result = match s.winner {
                        Some(Color::Black) => format!("黑胜 {:.1} 子", s.margin),
                        Some(Color::White) => format!("白胜 {:.1} 子", s.margin),
                        _ => "和棋".to_string(),
                    };
                    ui.label(
                        egui::RichText::new(result)
                            .size(15.0)
                            .strong()
                            .color(egui::Color32::from_rgb(235, 176, 106)),
                    );
                    ui.add_space(4.0);
                    if ui.button("清除判断").clicked() {
                        self.judged_score = None;
                    }
                });
            ui.add_space(6.0);
        }

        // 计分阶段：死子标记工具
        if self.game.status() == GameStatus::Scoring {
            ui.separator();
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("计分阶段：")
                    .size(14.0)
                    .strong()
                    .color(egui::Color32::from_rgb(235, 176, 106)),
            );
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(self.marking_dead, "标记死子")
                    .on_hover_text("点击棋盘上的棋子切换死子状态")
                    .clicked()
                {
                    self.marking_dead = !self.marking_dead;
                }
                if ui.button("确认计分").clicked() {
                    let s = self.game.confirm_score();
                    let msg = match s.winner {
                        Some(Color::Black) => format!("黑胜 {} 子", s.margin),
                        Some(Color::White) => format!("白胜 {} 子", s.margin),
                        _ => "和棋".to_string(),
                    };
                    self.message = format!("终局：{msg}");
                    self.trigger_effect(s.winner);
                    self.marking_dead = false;
                    self.hint_cache = None;
                }
                if ui.button("取消计分").clicked() {
                    self.game.undo();
                    self.message = "继续对弈".to_string();
                    self.marking_dead = false;
                }
            });
            let s = self.game.current_score();
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(format!(
                    "自动判死 {} 子（可手动修正）",
                    s.dead_removed
                ))
                .size(12.0)
                .color(egui::Color32::from_rgb(160, 150, 135)),
            );
            ui.add_space(4.0);
        }

        // 终局计分卡片
        if self.game.status() == GameStatus::Finished {
            ui.separator();
            ui.add_space(6.0);
            let s = self.game.current_score();
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(52, 44, 36))
                .rounding(egui::Rounding::same(8.0))
                .inner_margin(egui::Margin::same(12.0))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("终局计分（中国规则）")
                            .size(14.0)
                            .strong()
                            .color(egui::Color32::from_rgb(217, 160, 91)),
                    );
                    ui.add_space(6.0);
                    let row = |ui: &mut egui::Ui, name: &str, stones: usize, terr: usize, total: usize| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "{name}　{stones} 子 + {terr} 空 = {total}"
                                ))
                                .size(13.0),
                            );
                        });
                    };
                    row(ui, "黑方", s.black_stones, s.black_territory, s.black_total);
                    row(ui, "白方", s.white_stones, s.white_territory, s.white_total);
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(format!("黑贴 {:.2} 子", self.game.komi()))
                            .size(12.0)
                            .color(egui::Color32::from_rgb(160, 150, 135)),
                    );
                    ui.add_space(6.0);
                    let result = match s.winner {
                        Some(Color::Black) => format!("黑胜 {:.1} 子", s.margin),
                        Some(Color::White) => format!("白胜 {:.1} 子", s.margin),
                        _ => "和棋".to_string(),
                    };
                    ui.label(
                        egui::RichText::new(result)
                            .size(18.0)
                            .strong()
                            .color(egui::Color32::from_rgb(235, 176, 106)),
                    );
                    ui.add_space(4.0);
                    if ui.button("再来一局").clicked() {
                        self.new_game();
                    }
                });
        }
    }
}

/// 木纹渐变棋盘背景。
fn paint_wood_background(painter: &egui::Painter, rect: egui::Rect) {
    let rounding = egui::Rounding::same(10.0);
    painter.rect_filled(rect, rounding, egui::Color32::from_rgb(206, 160, 92));
    // 纵向渐变（上浅下深）
    let steps = 12;
    for i in 0..steps {
        let t0 = i as f32 / steps as f32;
        let t1 = (i + 1) as f32 / steps as f32;
        let y0 = rect.top() + rect.height() * t0;
        let y1 = rect.top() + rect.height() * t1;
        let shade = (i as f32 / steps as f32 * 30.0) as u8;
        let r = 206u8.saturating_sub(shade);
        let g = 160u8.saturating_sub(shade);
        let b = 92u8.saturating_sub(shade * 2);
        painter.rect_filled(
            egui::Rect::from_min_max(egui::pos2(rect.left(), y0), egui::pos2(rect.right(), y1)),
            egui::Rounding::ZERO,
            egui::Color32::from_rgb(r, g, b),
        );
    }
    // 木纹细线
    for i in 0..20 {
        let y = rect.top() + 8.0 + i as f32 * rect.height() / 21.0;
        painter.line_segment(
            [
                egui::pos2(rect.left() + 4.0, y),
                egui::pos2(rect.right() - 4.0, y + (i % 3) as f32 * 1.5),
            ],
            egui::Stroke::new(0.8, egui::Color32::from_rgba_unmultiplied(120, 85, 40, 26)),
        );
    }
    // 外边框
    painter.rect_stroke(
        rect,
        rounding,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(88, 58, 30)),
    );
}

/// 绘制立体棋子（阴影 + 渐变 + 高光）。
fn paint_stone(painter: &egui::Painter, center: egui::Pos2, r: f32, color: Color) {
    // 投影
    let shadow = egui::pos2(center.x + r * 0.08, center.y + r * 0.12);
    painter.circle_filled(
        shadow,
        r,
        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 55),
    );
    match color {
        Color::Black => {
            painter.circle_filled(center, r, egui::Color32::from_rgb(26, 26, 28));
            painter.circle_filled(
                center,
                r * 0.82,
                egui::Color32::from_rgb(52, 52, 56),
            );
            let hi = egui::pos2(center.x - r * 0.28, center.y - r * 0.34);
            painter.circle_filled(
                hi,
                r * 0.34,
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 60),
            );
        }
        Color::White => {
            painter.circle_filled(center, r, egui::Color32::from_rgb(232, 226, 214));
            painter.circle_filled(center, r * 0.88, egui::Color32::from_rgb(250, 246, 236));
            painter.circle_stroke(
                center,
                r,
                egui::Stroke::new(1.2, egui::Color32::from_rgb(170, 160, 142)),
            );
            let hi = egui::pos2(center.x - r * 0.28, center.y - r * 0.34);
            painter.circle_filled(
                hi,
                r * 0.30,
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 190),
            );
        }
        Color::Empty => {}
    }
}

/// 半透明预览棋子。
fn paint_stone_translucent(painter: &egui::Painter, center: egui::Pos2, r: f32, color: Color) {
    match color {
        Color::Black => {
            painter.circle_filled(
                center,
                r,
                egui::Color32::from_rgba_unmultiplied(30, 30, 34, 130),
            );
        }
        Color::White => {
            painter.circle_filled(
                center,
                r,
                egui::Color32::from_rgba_unmultiplied(245, 242, 232, 130),
            );
        }
        Color::Empty => {}
    }
}

/// 程序图标（16×16：木色底 + 黑白两子）。
fn make_icon() -> std::sync::Arc<egui::IconData> {
    const N: usize = 16;
    let mut rgba = Vec::with_capacity(N * N * 4);
    for y in 0..N {
        for x in 0..N {
            let (r, g, b) = if (x + y) % 6 < 3 {
                (206u8, 160u8, 92u8)
            } else {
                (196u8, 150u8, 82u8)
            };
            let (r, g, b) = if (x - 4) * (x - 4) + (y - 4) * (y - 4) <= 9 {
                (30, 30, 34)
            } else if (x - 11) * (x - 11) + (y - 11) * (y - 11) <= 9 {
                (246, 242, 232)
            } else {
                (r, g, b)
            };
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
    }
    std::sync::Arc::new(egui::IconData {
        rgba,
        width: N as u32,
        height: N as u32,
    })
}

/// 启动 GUI。
pub fn run_gui() -> eframe::Result<()> {
    // 神经网络模型只加载一次（内嵌权重，失败时界面自动回退经典引擎）
    if !crate::nn::init(None) {
        eprintln!("[提示] 神经网络模型加载失败，界面将使用经典引擎");
    }

    // 默认窗口：棋盘 600×600 + 右侧面板 280 + 顶栏底栏 ≈ 950×870
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([950.0, 870.0])
            .with_min_inner_size([760.0, 700.0])
            .with_title("GobanGo 围棋")
            .with_icon(make_icon()),
        ..Default::default()
    };

    eframe::run_native(
        "GobanGo 围棋",
        options,
        Box::new(|cc| {
            setup_fonts(&cc.egui_ctx);
            setup_style(&cc.egui_ctx);
            Ok(Box::new(GoApp::new()))
        }),
    )
}

/// 加载中文字体（Windows 微软雅黑，缺失时退回默认字体）。
fn setup_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let mut loaded = false;
    for path in [
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\simsun.ttc",
    ] {
        if let Ok(data) = std::fs::read(path) {
            fonts
                .font_data
                .insert("cjk".to_owned(), egui::FontData::from_owned(data));
            fonts
                .families
                .get_mut(&egui::FontFamily::Proportional)
                .unwrap()
                .insert(0, "cjk".to_owned());
            fonts
                .families
                .get_mut(&egui::FontFamily::Monospace)
                .unwrap()
                .push("cjk".to_owned());
            loaded = true;
            break;
        }
    }
    if !loaded {
        // macOS / Linux 常见中文字体兜底
        for path in [
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ] {
            if let Ok(data) = std::fs::read(path) {
                fonts
                    .font_data
                    .insert("cjk".to_owned(), egui::FontData::from_owned(data));
                fonts
                    .families
                    .get_mut(&egui::FontFamily::Proportional)
                    .unwrap()
                    .insert(0, "cjk".to_owned());
                break;
            }
        }
    }
    ctx.set_fonts(fonts);
}

/// 温暖深色主题样式。
fn setup_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.visuals.dark_mode = true;
    style.visuals.override_text_color = Some(egui::Color32::from_rgb(225, 218, 205));
    style.visuals.panel_fill = egui::Color32::from_rgb(42, 36, 30);
    style.visuals.window_fill = egui::Color32::from_rgb(42, 36, 30);
    style.visuals.extreme_bg_color = egui::Color32::from_rgb(30, 26, 22);
    style.visuals.faint_bg_color = egui::Color32::from_rgb(52, 44, 36);
    style.visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(62, 53, 44);
    style.visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(56, 48, 40);
    style.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(82, 68, 52);
    style.visuals.widgets.active.bg_fill = egui::Color32::from_rgb(140, 96, 46);
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(217, 160, 91));
    style.visuals.selection.bg_fill = egui::Color32::from_rgb(120, 84, 40);
    style.visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(235, 176, 106));
    let rounding = egui::Rounding::same(6.0);
    style.visuals.widgets.inactive.rounding = rounding;
    style.visuals.widgets.hovered.rounding = rounding;
    style.visuals.widgets.active.rounding = rounding;
    style.visuals.widgets.open.rounding = rounding;
    style.visuals.window_rounding = egui::Rounding::same(10.0);
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    ctx.set_style(style);
}

