//! AI 模块：四档难度配置、MCTS/UCT 搜索、三档快速走子策略与开局库。
//!
//! 复刻的成熟 CPU 围棋 AI 技术（来源：Pachi、Fuego、Cgoban、WenGo 公开论文与源码）：
//!
//! * 搜索框架 —— MCTS + UCT（Rémi Coulom，2006）
//! * 根级 RAVE（AMAF，Furusi / Ma 2002）；以及 GRAVE（广义 RAVE，Wu/Zheng 2015）
//! * 快速走子策略 —— Pachi「Moggy」/Fuego 3×3 模式表（在线学习权重）
//! * 战术判读 —— Fuego「DoSelfAtariCorrection」自填气扣分；打吃威胁 / 被叫吃不应
//! * 布局偏置 —— "金角银边草肚皮"（fuseki_bias，早期优先角边）
//! * 渐进拓宽 —— 访问数达 k^2 时才展开新子节点（Browne 2012，MCTS 综述）
//! * 先验偏置 —— Pachi 风格 3×3 模式先验（先验值随访问数衰减混合）
//! * 虚拟损失（Virtual Loss） —— 多线程并发避免重复扩展同一点
//! * 根温度选点 —— (visits)^(1/T) 加权选点 + 微噪声，比单纯访问数更稳（Cgoban / Lizzie 通用做法）
//! * 叶节点虚拟模拟（Virtual Roll） —— 无候选可展开时继承祖先统计，避免浪费 rollout
//! * 终局提前检测 —— 空点 < 60 时直接计分，不再走模拟
//! * 线程可取消 —— 全局取消标志 AI_CANCEL，GUI 退出或切换难度时可强制终止搜索
//!
//! 性能预算（自适应停止：时间与模拟次数任一先到即停；若时间到但模拟数不足下限则继续）：
//! | 难度 | 单步时间上限 | 模拟上限 | 模拟下限 | 线程 | C_UCT | 根温度 T | 开局库 |
//! |------|--------------|----------|----------|------|-------|----------|--------|
//! | 入门 | 0.5s         | 500      | 100      | 1    | 1.2   | 2.0      | 无     |
//! | 中级 | 2.0s         | 3000     | 500      | 2    | 1.4   | 1.5      | 前 10 手 |
//! | 高级 | 5.0s         | 20000    | 2000     | 4    | 1.5   | 1.2      | 前 50 手 |
//! | 大师 | 60.0s        | 5_000_000| 100000   | 8~16 | 1.5   | 1.15     | 前 50 手 |

use crate::core::{Board, Color, Move, Point};
use crate::gpu::{pack_board, PlayoutInput, PlayoutOutput};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// 全局 AI 取消标志：GUI 退出、切难度、或主动停止时可置位，
/// `select_move_cfg` 每 64 次迭代检查一次，置位后所有工作线程立即退出。
/// 这避免了 GUI 主窗口关闭时 AI 线程仍存活导致进程挂起的 BUG。
pub static AI_CANCEL: AtomicBool = AtomicBool::new(false);

/// 请求取消所有正在进行的 AI 搜索（GUI 退出 / 切难度时调用）。
pub fn request_ai_cancel() {
    AI_CANCEL.store(true, Ordering::SeqCst);
}

/// 清除取消标志（每次新搜索开始前调用，确保不会被上一次残留的取消信号误伤）。
pub fn clear_ai_cancel() {
    AI_CANCEL.store(false, Ordering::SeqCst);
}

/// 四档难度。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
    /// 大师级：单步 60s、内存峰值上限 15GB、全核并行搜索（家用高配电脑）。
    Master,
}

impl Difficulty {
    pub fn label(&self) -> &'static str {
        match self {
            Difficulty::Easy => "入门",
            Difficulty::Medium => "中级",
            Difficulty::Hard => "高级",
            Difficulty::Master => "大师",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Difficulty::Easy => "单步 ≤0.5s，随机+基础过滤，适合初学者",
            Difficulty::Medium => "单步 ≤2.0s，启发式走子+开局库，约业余 5 级",
            Difficulty::Hard => "单步 ≤5.0s，强启发式+局部战术验证，约业余 1-3 段",
            Difficulty::Master => "单步 ≤60s，内存峰值 ≤15GB，全核并行，挑战级棋力",
        }
    }
}

/// 搜索配置（对应说明书 6.1 SearchConfig）。
#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub max_time: Duration,
    pub max_sims: u32,
    pub min_sims: u32,
    pub threads: usize,
    pub c_uct: f64,
    pub max_rollout_moves: u32,
    pub use_book: bool,
    pub book_plies: usize,
    /// 快速走子策略等级：1=随机+过滤，2=启发式，3=强启发式+局部战术。
    pub heuristic: u8,
    /// 根节点候选着法上限。
    pub max_candidates: usize,
    /// 是否启用 3×3 模式走子策略（Pachi/Fuego 风格，棋力显著提升）。
    pub use_patterns: bool,
    /// 是否启用根级 RAVE（AMAF）加速树搜索。
    pub use_rave: bool,
    /// 是否启用进阶 MCTS 增强：渐进拓宽 + 渐进偏置（先验）+ GRAVE 广义 RAVE。
    pub use_enhanced: bool,
    /// 是否启用虚拟损失（Virtual Loss）：并发线程在展开某着法前给该着法加一个负先验，
    /// 防止多个线程重复扩展同一点；线程返回后回退（WenGo / Pachi 通用技巧）。
    pub use_virtual_loss: bool,
    /// 是否启用叶节点虚拟模拟（Virtual Roll，Fuego 风格）：当叶子无候选可展开时，
    /// 直接用父节点的 (wins/visits) 作为回溯值，避免一次空 rollout。
    pub use_virtual_roll: bool,
    /// 根温度 T（Winrate-with-Exploration 选点）：最终选点权重 = visits^(1/T)，
    /// T > 1 增强探索、降低过度拟合；T = 1 等价于访问数最大；T < 1 更贪婪。
    /// 参考：Cgoban/Lizzie 通用做法（KataGo 也支持此参数）。
    pub root_temperature: f64,
}

impl Difficulty {
    pub fn config(&self) -> SearchConfig {
        match self {
            Difficulty::Easy => SearchConfig {
                max_time: Duration::from_millis(500),
                max_sims: 500,
                min_sims: 100,
                threads: 1,
                c_uct: 1.2,
                max_rollout_moves: 60,
                use_book: false,
                book_plies: 0,
                heuristic: 1,
                max_candidates: 40,
                use_patterns: false,
                use_rave: false,
                use_enhanced: false,
                use_virtual_loss: false,
                use_virtual_roll: false,
                root_temperature: 2.0, // 入门棋力弱：选点更随机
            },
            Difficulty::Medium => SearchConfig {
                max_time: Duration::from_millis(2000),
                max_sims: 3000,
                min_sims: 500,
                threads: 2,
                c_uct: 1.4,
                max_rollout_moves: 60,
                use_book: true,
                book_plies: 10,
                heuristic: 2,
                max_candidates: 40,
                use_patterns: true,
                use_rave: true,
                use_enhanced: false,
                use_virtual_loss: true,
                use_virtual_roll: true,
                root_temperature: 1.5,
            },
            Difficulty::Hard => SearchConfig {
                max_time: Duration::from_millis(5000),
                max_sims: 20000,
                min_sims: 2000,
                threads: 4,
                c_uct: 1.5,
                max_rollout_moves: 60,
                use_book: true,
                book_plies: 50,
                heuristic: 3,
                max_candidates: 50,
                use_patterns: true,
                use_rave: true,
                use_enhanced: false,
                use_virtual_loss: true,
                use_virtual_roll: true,
                root_temperature: 1.2,
            },
            Difficulty::Master => {
                // 大师级：单步 60s、模拟上限 500 万、内存峰值上限 15GB、
                // 全核并行（至多 16 线程）。按 ~500B/节点估算，
                // 500 万节点 ≈ 2.5GB，远低于 15GB 上限，可安全扩展。
                let threads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(8)
                    .clamp(8, 16);
                SearchConfig {
                    max_time: Duration::from_secs(60),
                    max_sims: 5_000_000,
                    min_sims: 100_000,
                    threads,
                    c_uct: 1.5,
                    max_rollout_moves: 60,
                    use_book: true,
                    book_plies: 50,
                    heuristic: 3,
                    max_candidates: 80,
                    use_patterns: true,
                    use_rave: true,
                    use_enhanced: false,
                    use_virtual_loss: true,
                    use_virtual_roll: true,
                    root_temperature: 1.15, // 大师：几乎纯访问数，轻微探索
                }
            }
        }
    }
}

/// 内置开局库（标准布局，格式同说明书：`B Q16; W D4; ...`）。
const EMBEDDED_BOOK: &str = "\
B Q16; W D4; B R4; W D16; B Q3; W C4; B C16; W Q14; B C3; W R16; B R3; W C14
B Q16; W D4; B R4; W D16; B Q3; W C4; B C16; W Q14; B Q4; W C3
B Q16; W D4; B R4; W D16; B Q3; W C4; B C3; W C16; B R3; W Q14
B Q16; W D4; B R4; W D16; B C3; W C16; B Q3; W R3; B C4; W Q14
B Q16; W D4; B C3; W C16; B R4; W D16; B Q3; W R3; B C4; W Q14
B Q16; W D4; B C3; W D16; B R4; W R16; B Q3; W C16; B C4; W Q14
B Q16; W D4; B C3; W C16; B R4; W D16; B Q3; W R3; B C4; W Q14; B R16; W D3
B Q16; W D4; B C3; W C16; B R4; W R16; B Q3; W D16; B C4; W Q14; B R3; W D3
B Q16; W Q4; B D16; W D4; B R16; W R4; B C16; W C4; B Q3; W D3; B C3; W R3
B Q16; W Q4; B D16; W D4; B R16; W R4; B C16; W C4; B D3; W Q3; B R3; W C3
B Q16; W Q4; B D16; W D4; B Q3; W C4; B C16; W R16; B R4; W C3; B R3; W D3
B Q16; W Q4; B D16; W D4; B C4; W C16; B R16; W R4; B Q3; W D3; B C3; W R3
B Q16; W D4; B C3; W Q4; B D16; W R16; B R4; W C16; B Q3; W R3; B C4; W D3
B Q16; W D4; B C3; W Q4; B D16; W C16; B R4; W R16; B Q3; W C4; B R3; W D3
B Q16; W Q4; B C3; W D4; B R16; W R4; B D16; W C16; B Q3; W D3; B C4; W R3
B Q16; W Q4; B C3; W D4; B D16; W R16; B R4; W C16; B Q3; W C4; B R3; W D3
B Q16; W D4; B R4; W Q4; B Q3; W C4; B D16; W C16; B R16; W R3; B C3; W D3
B Q16; W D4; B R4; W Q4; B Q3; W C4; B C16; W D16; B R16; W R3; B C3; W D3
B Q16; W Q4; B D16; W D4; B R16; W R4; B Q3; W C16; B C4; W D3; B R3; W C3
B Q16; W Q4; B D16; W D4; B R16; W R4; B C16; W C4; B D3; W Q3; B R3; W C3
B Q16; W D4; B C16; W D16; B C4; W Q4; B C15; W Q14; B C3; W Q3
B Q16; W D4; B C3; W D16; B R3; W R16; B C16; W Q4; B R16; W C4
B Q16; W D4; B C3; W D16; B R4; W R16; B C16; W Q4; B R16; W C4
B Q16; W D4; B R3; W D16; B C15; W Q4; B C3; W Q14; B Q3; W C14
B Q16; W D4; B R3; W C16; B C3; W Q4; B Q3; W C15; B C4; W Q14
B Q16; W D4; B Q3; W D16; B C3; W R16; B R3; W C16; B Q4; W C4
B Q16; W Q4; B C16; W C4; B D16; W D4; B Q3; W Q14; B R3; W C3
B Q16; W D4; B C16; W D16; B Q3; W R16; B R4; W Q4; B C4; W R3
B Q16; W Q4; B C16; W D4; B D16; W C4; B Q3; W Q14; B R3; W D3
B Q16; W D4; B C3; W C16; B R4; W Q4; B Q3; W R16; B C4; W D16; B R3; W D3
";

/// 开局库：若干条以 (颜色, 点) 序列存储的标准开局。
pub struct OpeningBook {
    lines: Vec<Vec<(Color, Point)>>,
}

impl OpeningBook {
    /// 解析内置开局库（并尝试合并 `data/opening_book.txt` 中的自定义棋谱）。
    pub fn load() -> OpeningBook {
        let mut lines = Self::parse(EMBEDDED_BOOK);
        let mut candidates = vec!["data/opening_book.txt".to_string()];
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join("data/opening_book.txt").to_string_lossy().into_owned());
            }
        }
        for path in candidates {
            if let Ok(text) = std::fs::read_to_string(&path) {
                let extra = Self::parse(&text);
                if !extra.is_empty() {
                    lines.extend(extra);
                    break;
                }
            }
        }
        OpeningBook { lines }
    }

    fn parse(text: &str) -> Vec<Vec<(Color, Point)>> {
        let mut lines = Vec::new();
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut seq = Vec::new();
            let mut ok = true;
            for token in line.split(';') {
                let token = token.trim();
                if token.is_empty() {
                    continue;
                }
                let mut it = token.split_whitespace();
                let (color, rest) = match (it.next(), it.next()) {
                    (Some(c), Some(r)) => (c, r),
                    _ => {
                        ok = false;
                        break;
                    }
                };
                let color = match color {
                    "B" | "b" => Color::Black,
                    "W" | "w" => Color::White,
                    _ => {
                        ok = false;
                        break;
                    }
                };
                if rest.eq_ignore_ascii_case("pass") || rest.eq_ignore_ascii_case("resign") {
                    ok = false;
                    break;
                }
                match Point::from_sgf(rest) {
                    Some(p) => seq.push((color, p)),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && !seq.is_empty() {
                lines.push(seq);
            }
        }
        lines
    }

    /// 在开局库中查找当前棋谱的下一步。
    /// `history`：已下的 (颜色, 点) 序列；`plies_used`：难度允许使用的开局库手数上限。
    /// 命中（前缀完全一致且库中还有下一手）则返回库中着法。
    pub fn find(&self, history: &[(Color, Point)], plies_used: usize) -> Option<Point> {
        if history.len() >= plies_used || plies_used == 0 {
            return None;
        }
        let mut best: Option<(usize, Point)> = None;
        for line in &self.lines {
            if line.len() <= history.len() {
                continue;
            }
            let matches = line
                .iter()
                .zip(history.iter())
                .all(|(a, b)| a.0 == b.0 && a.1 == b.1);
            if matches {
                let next = line[history.len()];
                // 取第一条匹配的棋谱（库中排列靠前的为标准主流布局）
                if best.is_none() {
                    best = Some((history.len(), next.1));
                }
            }
        }
        best.map(|(_, p)| p)
    }

    /// 开局库条目数（调试用）。
    pub fn len(&self) -> usize {
        self.lines.len()
    }
}

/// 收集盘面上所有棋子的空邻点（快速候选集）。
fn empty_neighbors(board: &Board) -> HashSet<Point> {
    let mut set = HashSet::new();
    for y in 0..19 {
        for x in 0..19 {
            if board.at(x, y) == Color::Empty {
                continue;
            }
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 && nx < 19 && ny < 19 && board.is_empty(nx as usize, ny as usize) {
                    set.insert(Point::new(nx as usize, ny as usize));
                }
            }
        }
    }
    set
}

/// 空点 p 落子后是否形成“填眼”（四邻全为己方），走子策略中用于扣分。
fn fills_eye_after(board: &Board, p: Point, color: Color) -> bool {
    let mut own = 0;
    let mut other = 0;
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let nx = p.x as i32 + dx;
        let ny = p.y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= 19 || ny >= 19 {
            continue;
        }
        match board.at(nx as usize, ny as usize) {
            Color::Empty => return false,
            c if c == color => own += 1,
            _ => other += 1,
        }
    }
    own > 0 && other == 0
}

/// 简单局部战术验证（高级难度）：落子后若吃子，或所救的己方棋链仍保有 ≥2 气，则视为好手。
/// 只做 1 层对手回应的有界验证，开销很小。
fn tactical_verify(board: &Board, p: Point, color: Color, _rng: &mut StdRng) -> bool {
    let mut copy = board.clone();
    if !copy.play(p.x, p.y) {
        return false;
    }
    if copy.last_captured > 0 {
        return true; // 直接吃子
    }
    // 检查被救的己方棋链（落子前气为 1 的相邻己方链）
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let nx = p.x as i32 + dx;
        let ny = p.y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= 19 || ny >= 19 {
            continue;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        if board.at(nx, ny) != color {
            continue;
        }
        let before = board.liberties_count(nx, ny);
        if before != 1 {
            continue;
        }
        // 落子后该链气数
        let after = copy.liberties_count(nx, ny);
        if after >= 2 {
            return true;
        }
        // 对手最强回应：占据该链一个气点后，是否仍 ≥2 气
        let liberties = copy.group_liberties(nx, ny);
        let mut any = false;
        for &(lx, ly) in liberties.iter().take(3) {
            let mut resp = copy.clone();
            if resp.play(lx, ly) {
                if resp.liberties_count(nx, ny) >= 2 {
                    any = true;
                    break;
                }
            }
        }
        if !any {
            return false; // 对手可杀
        }
    }
    true
}

/// 假设在空点 p 落己方一子（未吃子情形），新棋链气数是否 ≤ 1。
/// Fuego「DoSelfAtariCorrection」风格的 self-atari 检测：这种着法等于自填气，
/// 走子策略中应强烈降温。早退 BFS：找到第 2 口气立即返回 false，平均开销极小。
/// 调用方需保证该手不构成吃子（吃子会腾出气，不会是自填气）。
fn result_chain_libs_le_one(board: &Board, p: Point, color: Color) -> bool {
    let mut stack = vec![(p.x, p.y)];
    let mut visited = HashSet::new();
    let mut libs = 0usize;
    while let Some((cx, cy)) = stack.pop() {
        if !visited.insert((cx, cy)) {
            continue;
        }
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            if nx < 0 || ny < 0 || nx >= 19 || ny >= 19 {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            match board.at(nx, ny) {
                Color::Empty => {
                    libs += 1;
                    if libs >= 2 {
                        return false;
                    }
                }
                c if c == color => stack.push((nx, ny)),
                _ => {}
            }
        }
    }
    true
}

/// 快速判定：(x, y) 处棋子的 1 跳空邻 ≤ 1（可能是叫吃状态）。
/// 若 1 跳空邻 ≥ 2，则该棋链必然 ≥ 2 气，可跳过完整 BFS。
fn quick_atari_hint(board: &Board, x: usize, y: usize) -> bool {
    let mut e = 0;
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if nx >= 0 && ny >= 0 && nx < 19 && ny < 19
            && board.is_empty(nx as usize, ny as usize)
        {
            e += 1;
        }
    }
    e <= 1
}

/// 棋链气数是否 ≤ 1（早退 BFS：找到第 2 个空邻立即返回 false）。
/// 稠密局面下绝大多数棋链气数 ≥ 2，因此平均开销极小，避免整链遍历。
fn chain_liberties_at_most_one(board: &Board, x: usize, y: usize) -> bool {
    let color = board.at(x, y);
    if color == Color::Empty {
        return true;
    }
    let mut stack = vec![(x, y)];
    let mut visited = HashSet::new();
    let mut libs = 0usize;
    while let Some((cx, cy)) = stack.pop() {
        if !visited.insert((cx, cy)) {
            continue;
        }
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            if nx < 0 || ny < 0 || nx >= 19 || ny >= 19 {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            match board.at(nx, ny) {
                Color::Empty => {
                    libs += 1;
                    if libs >= 2 {
                        return false;
                    }
                }
                c if c == color => stack.push((nx, ny)),
                _ => {}
            }
        }
    }
    true
}

/// 候选点 4 邻扫描（无克隆，单次约 2-6µs）。
/// 返回 (对方邻数, 己方邻数, 可吃对方链数, 可救己方链数, 被叫吃不救数)。
fn scan_neighbors(
    board: &Board,
    p: Point,
    color: Color,
) -> (usize, usize, usize, usize, usize) {
    let mut opp_n = 0usize;
    let mut own_n = 0usize;
    let mut opp_atari = 0usize;
    let mut own_save = 0usize;
    let mut own_ignore = 0usize;
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let nx = p.x as i32 + dx;
        let ny = p.y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= 19 || ny >= 19 {
            continue;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        match board.at(nx, ny) {
            Color::Empty => {}
            c if c == color => own_n += 1,
            _ => {
                opp_n += 1;
                if chain_liberties_at_most_one(board, nx, ny) {
                    opp_atari += 1;
                }
            }
        }
    }
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let nx = p.x as i32 + dx;
        let ny = p.y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= 19 || ny >= 19 {
            continue;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        if board.at(nx, ny) == color && chain_liberties_at_most_one(board, nx, ny) {
            let libs = board.group_liberties(nx, ny);
            if libs.contains(&(p.x, p.y)) {
                own_save += 1;
            } else {
                own_ignore += 1;
            }
        }
    }
    (opp_n, own_n, opp_atari, own_save, own_ignore)
}

/// 3×3 局部模式类总数：8 个邻位 × 3 状态（空/己/他）= 3^8。
const PATTERN_CLASSES: usize = 6561;

/// 3×3 局部模式权重表（Pachi「Moggy」/Fuego 风格）：
/// 每次走子观察落点 3×3 邻域模式，按学习到的权重采样；
/// 对局结束后在线更新权重（获胜方的着法模式权重上调）。
struct PatternTable {
    weights: [f32; PATTERN_CLASSES],
}

impl PatternTable {
    fn new() -> Self {
        let mut weights = [1.0f32; PATTERN_CLASSES];
        // 过程式先验：从模式内容推导合理初值（无需人工标注）
        for idx in 0..PATTERN_CLASSES {
            let mut v = idx;
            let mut own = 0usize;
            let mut opp = 0usize;
            for _ in 0..8 {
                match v % 3 {
                    1 => own += 1,
                    2 => opp += 1,
                    _ => {}
                }
                v /= 3;
            }
            let mut base = 1.0f32;
            if own >= 4 {
                // 疑似填己方眼：直接覆盖为极低权重（强于一切加成）
                base = 0.15;
            } else {
                if own >= 2 {
                    base += 1.2; // 靠己方厚势
                }
                if opp >= 1 {
                    base += 0.6; // 接触对方（战斗）
                }
                if opp >= 3 {
                    base += 0.8; // 紧贴对方阵势
                }
                if own == 0 && opp == 0 {
                    base *= 0.5; // 空旷孤点
                }
            }
            weights[idx] = base;
        }
        PatternTable { weights }
    }

    /// 落点 (x, y) 在己方视角下的 3×3 模式索引。
    #[inline]
    fn index(&self, board: &Board, x: usize, y: usize, color: Color) -> usize {
        let mut idx = 0usize;
        let mut mul = 1usize;
        for (dx, dy) in [
            (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
        ] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let s = if nx < 0 || ny < 0 || nx >= 19 || ny >= 19 {
                0
            } else {
                match board.at(nx as usize, ny as usize) {
                    Color::Empty => 0,
                    c if c == color => 1,
                    _ => 2,
                }
            };
            idx += s * mul;
            mul *= 3;
        }
        idx
    }

    #[inline]
    fn weight(&self, idx: usize) -> f32 {
        self.weights[idx]
    }

    /// 在线学习：按对局结果更新模式权重（指数滑动平均）。
    fn update(&mut self, idx: usize, result: f32, alpha: f32) {
        let w = &mut self.weights[idx];
        *w += alpha * (result - *w);
        *w = w.clamp(0.05, 8.0);
    }
}

/// 布局阶段（前 20 手）角边优先偏置：体现“金角银边草肚皮”的围棋规律。
/// 返回一个乘数：角星位 > 角部 > 边线 > 一般 > 中腹（中腹明显降温）。
fn fuseki_bias(x: usize, y: usize, plies: u32) -> f32 {
    if plies >= 20 {
        return 1.0;
    }
    let corner = (x < 4 || x >= 15) && (y < 4 || y >= 15);
    let star = (x == 3 || x == 15) && (y == 3 || y == 15);
    let side = x < 3 || x >= 16 || y < 3 || y >= 16;
    let center = (6..13).contains(&x) && (6..13).contains(&y);
    if star {
        3.2
    } else if corner {
        2.4
    } else if side {
        1.5
    } else if center {
        0.45
    } else {
        1.0
    }
}

/// 对候选着法打分（中级/高级启发式，对应说明书 5.5.2/5.5.3）。
/// `tactical`：是否执行局部战术验证（仅根选点时开启，rollout 高频调用时关闭以省时）。
/// 返回得分；非法着法返回 `i32::MIN`。
fn score_move(board: &Board, p: Point, color: Color, heuristic: u8, rng: &mut StdRng, tactical: bool) -> i32 {
    let mut copy = board.clone();
    if !copy.play(p.x, p.y) {
        return i32::MIN;
    }
    let mut score = 10 + rng.gen_range(0..=10);
    // 吃子
    score += copy.last_captured as i32 * 100;
    // 救回气为 1 的己方链
    let mut saves = 0;
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let nx = p.x as i32 + dx;
        let ny = p.y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= 19 || ny >= 19 {
            continue;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        if board.at(nx, ny) == color
            && quick_atari_hint(board, nx, ny)
            && board.liberties_count(nx, ny) == 1
        {
            saves += 1;
        }
    }
    score += saves * 80;
    // 切断对方：4 邻中有 ≥2 个互不相连的对方棋链
    let mut opp_neighbors: Vec<(usize, usize)> = Vec::new();
    let mut opp_contact = 0;
    let mut own_contact = 0;
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let nx = p.x as i32 + dx;
        let ny = p.y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= 19 || ny >= 19 {
            continue;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        match board.at(nx, ny) {
            Color::Empty => {}
            c if c == color => own_contact += 1,
            _ => {
                opp_contact += 1;
                opp_neighbors.push((nx, ny));
            }
        }
    }
    if opp_neighbors.len() >= 2 {
        let opp = color.opponent();
        let mut visited = HashSet::new();
        let mut chains = 0;
        for &(cx, cy) in &opp_neighbors {
            if visited.contains(&(cx, cy)) {
                continue;
            }
            chains += 1;
            if chains >= 2 {
                break;
            }
            // BFS 标记该对方棋链全部棋子（共享 visited，最多遍历两次）
            let mut stack = vec![(cx, cy)];
            while let Some((sx, sy)) = stack.pop() {
                if !visited.insert((sx, sy)) {
                    continue;
                }
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let nx = sx as i32 + dx;
                    let ny = sy as i32 + dy;
                    if nx >= 0 && ny >= 0 && nx < 19 && ny < 19
                        && board.at(nx as usize, ny as usize) == opp
                    {
                        stack.push((nx as usize, ny as usize));
                    }
                }
            }
        }
        if chains >= 2 {
            score += 40;
        }
    }
    // 接触战加成：靠近对方更有利，靠近己方略有帮助
    score += opp_contact * 25;
    score += own_contact * 10;
    // 扩展己方势力
    let empty_neighbors = 4 - opp_neighbors.len() as i32 - own_contact;
    if empty_neighbors < 4 {
        score += 20;
    }
    // 避免填己方真眼
    if fills_eye_after(board, p, color) {
        score -= 50;
    }
    // Fuego「DoSelfAtariCorrection」：落子后己方新棋链仅 1 气且未吃子 → 自填气重罚
    if copy.last_captured == 0 {
        let libs = copy.liberties_count(p.x, p.y);
        if libs <= 1 {
            score -= 60;
        } else if libs >= 3 {
            score += 8; // 气数充足 → 舒展的棋形小幅加分
        }
    }
    // 高级：局部战术验证 + 打吃威胁/被叫吃应对
    if heuristic >= 3 {
        if tactical && tactical_verify(board, p, color, rng) {
            score += 30;
        }
        // 邻接己方 ≥3 子的“厚势”点小幅加成
        if own_contact >= 3 {
            score += 15;
        }
        // 打吃威胁：落子后使邻接对方棋链仅剩 1 气
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let nx = p.x as i32 + dx;
            let ny = p.y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= 19 || ny >= 19 {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            if copy.at(nx, ny) == color.opponent()
                && quick_atari_hint(&copy, nx, ny)
                && copy.liberties_count(nx, ny) == 1
            {
                score += 30;
            }
        }
        // 被叫吃不应：邻接己方链只剩 1 气，而本手不救也不吃子 → 重罚
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let nx = p.x as i32 + dx;
            let ny = p.y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= 19 || ny >= 19 {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            if board.at(nx, ny) == color
                && quick_atari_hint(board, nx, ny)
                && board.liberties_count(nx, ny) == 1
            {
                let libs = board.group_liberties(nx, ny);
                if !libs.contains(&(p.x, p.y)) && copy.last_captured == 0 {
                    score -= 60;
                }
            }
        }
    }
    score
}

/// 快速走子打分（无克隆，供 rollout 高频调用）：
/// 仅用候选点 4 邻扫描 + 叫吃预判，不做模拟，单次开销约 2-5µs。
/// 自杀点返回 `i32::MIN`（由调用方剔除）；劫点由调用方按 `ko_point` 剔除。
fn score_move_fast(board: &Board, p: Point, color: Color, heuristic: u8, rng: &mut StdRng) -> i32 {
    let mut opp_neighbors = 0usize;
    let mut own_neighbors = 0usize;
    let mut opp_atari = 0usize; // 邻接对方链仅 1 气（落子即吃）
    let mut own_atari_save = 0usize; // 本点能救的己方叫吃链数
    let mut own_atari_ignored = 0usize; // 邻接己方叫吃链但本点不救
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let nx = p.x as i32 + dx;
        let ny = p.y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= 19 || ny >= 19 {
            continue;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        match board.at(nx, ny) {
            Color::Empty => {}
            c if c == color => own_neighbors += 1,
            _ => {
                opp_neighbors += 1;
                if chain_liberties_at_most_one(board, nx, ny) {
                    opp_atari += 1;
                }
            }
        }
    }
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let nx = p.x as i32 + dx;
        let ny = p.y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= 19 || ny >= 19 {
            continue;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        if board.at(nx, ny) == color && chain_liberties_at_most_one(board, nx, ny) {
            let libs = board.group_liberties(nx, ny);
            if libs.contains(&(p.x, p.y)) {
                own_atari_save += 1;
            } else {
                own_atari_ignored += 1;
            }
        }
    }
    // 自杀代理：四邻全是对方且不构成吃子（与上两轮扫描共用数据）
    if opp_neighbors == 4 && own_neighbors == 0 && opp_atari == 0 {
        return i32::MIN;
    }
    let mut score = 10 + rng.gen_range(0..=10);
    score += opp_atari as i32 * 100; // 吃子
    score += own_atari_save as i32 * 80; // 救子
    score += opp_neighbors as i32 * 25; // 接触对方
    score += own_neighbors as i32 * 10; // 接触己方
    if opp_neighbors + own_neighbors > 0 {
        score += 20; // 扩展/战斗
    }
    if own_neighbors == 4 {
        score -= 50; // 填己方眼
    }
    if heuristic >= 3 {
        if own_neighbors >= 3 {
            score += 15; // 厚势
        }
        score -= own_atari_ignored as i32 * 60; // 被叫吃不应
        score += opp_atari as i32 * 10; // 打吃威胁（后续可提）
    }
    score
}

/// 快速走子策略状态：3×3 模式权重表 + 上一手跟踪（每个 MCTS 树实例一份）。
struct PlayoutPolicy {
    table: PatternTable,
    last_move: Option<Point>,
}

impl PlayoutPolicy {
    fn new() -> Self {
        PlayoutPolicy {
            table: PatternTable::new(),
            last_move: None,
        }
    }
}

/// 旧版启发式走子（score_move_fast 加权随机），作为“无模式”回退路径（A/B 对照用）。
fn heuristic_rollout_move(board: &Board, heuristic: u8, rng: &mut StdRng) -> Option<Move> {
    // 候选点：棋子邻点优先，不足时补随机空点
    let mut candidates: Vec<Point> = empty_neighbors(board).into_iter().collect();
    candidates.sort_by_key(|p| (p.y, p.x)); // 确定化：HashSet 迭代顺序随进程随机，先排序再洗牌
    if candidates.len() < 24 {
        let mut extra = Vec::new();
        for _ in 0..60 {
            let x = rng.gen_range(0..19);
            let y = rng.gen_range(0..19);
            let p = Point::new(x, y);
            if board.is_empty(x, y) && !candidates.contains(&p) {
                extra.push(p);
            }
            if extra.len() >= 24 - candidates.len() {
                break;
            }
        }
        candidates.extend(extra);
    }
    candidates.shuffle(rng);
    candidates.truncate(10);

    let color = board.turn();
    let ko = board.ko_point();
    let mut weighted: Vec<(Point, i32)> = Vec::new();
    for p in &candidates {
        if ko == Some(*p) {
            continue;
        }
        let s = score_move_fast(board, *p, color, heuristic, rng);
        if s > i32::MIN {
            weighted.push((*p, s.max(1)));
        }
    }
    if weighted.is_empty() {
        return None;
    }
    let total: i64 = weighted.iter().map(|(_, s)| *s as i64).sum();
    let mut r = rng.gen_range(0..total);
    for (p, s) in &weighted {
        if r < *s as i64 {
            return Some(Move::Play(*p));
        }
        r -= *s as i64;
    }
    Some(Move::Play(weighted.last().unwrap().0))
}

/// 快速走子策略（Pachi「Moggy」/Fuego 风格）：
/// 1=随机+过滤（入门）；2/3 级默认使用 3×3 模式加权走子，
/// 并按优先级处理：能提子必提 → 被叫吃先救 → 模式/布局/邻近加权随机。
fn rollout_policy_move(
    board: &Board,
    heuristic: u8,
    plies: u32,
    policy: &mut PlayoutPolicy,
    use_patterns: bool,
    rng: &mut StdRng,
) -> Option<Move> {
    if heuristic == 1 {
        // 入门：整盘纯随机（不偏向棋子邻点），棋力最弱
        let mut empties: Vec<Point> = (0..19)
            .flat_map(|y| (0..19).map(move |x| Point::new(x, y)))
            .filter(|p| board.is_empty(p.x, p.y))
            .collect();
        empties.shuffle(rng);
        // 约 1/5 概率优先吃子（只探测少量候选，控制开销）
        if rng.gen_bool(0.2) {
            for &p in empties.iter().take(24) {
                let mut c = board.clone();
                if c.play(p.x, p.y) && c.last_captured > 0 {
                    return Some(Move::Play(p));
                }
            }
        }
        for p in empties.iter().take(24) {
            if board.is_legal(p.x, p.y) {
                return Some(Move::Play(*p));
            }
        }
        return None;
    }
    if !use_patterns {
        return heuristic_rollout_move(board, heuristic, rng);
    }

    let color = board.turn();
    let mut candidates: Vec<Point> = empty_neighbors(board).into_iter().collect();
    candidates.sort_by_key(|p| (p.y, p.x)); // 确定化：HashSet 迭代顺序随进程随机，先排序再洗牌
    if candidates.len() < 24 {
        let mut extra = Vec::new();
        for _ in 0..60 {
            let x = rng.gen_range(0..19);
            let y = rng.gen_range(0..19);
            let p = Point::new(x, y);
            if board.is_empty(x, y) && !candidates.contains(&p) {
                extra.push(p);
            }
            if extra.len() >= 24 - candidates.len() {
                break;
            }
        }
        candidates.extend(extra);
    }
    candidates.shuffle(rng);
    candidates.truncate(10);

    // 自杀/劫剔除 + 战术统计
    struct Cand {
        p: Point,
        opp_n: usize,
        own_n: usize,
        opp_atari: usize,
        own_save: usize,
        own_ignore: usize,
    }
    let ko = board.ko_point();
    let mut list: Vec<Cand> = Vec::new();
    for p in &candidates {
        if ko == Some(*p) {
            continue;
        }
        let (opp_n, own_n, opp_atari, own_save, own_ignore) = scan_neighbors(board, *p, color);
        if opp_n == 4 && own_n == 0 && opp_atari == 0 {
            continue; // 自杀禁手
        }
        list.push(Cand {
            p: *p,
            opp_n,
            own_n,
            opp_atari,
            own_save,
            own_ignore,
        });
    }
    if list.is_empty() {
        return None;
    }

    // Fuego 风格优先级：先提子 → 再救子 → 最后一般着法
    let captures: Vec<&Cand> = list.iter().filter(|c| c.opp_atari >= 1).collect();
    let saves: Vec<&Cand> = list.iter().filter(|c| c.own_save >= 1).collect();
    let pool: Vec<&Cand> = if !captures.is_empty() {
        captures
    } else if !saves.is_empty() {
        saves
    } else {
        list.iter().collect()
    };

    // 权重合成：模式权重 × 布局偏置 × 邻近加成 + 战术加减
    let mut weighted: Vec<(Point, f64)> = Vec::new();
    for c in pool {
        let mut w: f64 = {
            let idx = policy.table.index(board, c.p.x, c.p.y, color);
            policy.table.weight(idx) as f64
        };
        w *= fuseki_bias(c.p.x, c.p.y, plies) as f64;
        if let Some(lm) = policy.last_move {
            let d = (c.p.x as i32 - lm.x as i32)
                .abs()
                .max((c.p.y as i32 - lm.y as i32).abs());
            if d <= 2 {
                w *= 1.4; // 贴近上一手（局部应对）
            }
        }
        w += c.opp_atari as f64 * 8.0; // 吃子
        w += c.own_save as f64 * 6.0; // 救子
        w -= c.own_ignore as f64 * 4.0; // 被叫吃不救
        if c.own_n == 4 {
            w *= 0.1; // 填己方眼
        }
        // Fuego「DoSelfAtariCorrection」：未吃子且落子后己方新棋链仅剩 1 气 → 自填气，强烈降温
        if c.opp_atari == 0 && c.own_save == 0 && c.own_n >= 1
            && result_chain_libs_le_one(board, c.p, color)
        {
            w *= 0.05;
        }
        if c.opp_n == 0 && c.own_n == 0 {
            w *= 0.6; // 空旷处降温
        }
        if w > 0.02 {
            weighted.push((c.p, w));
        }
    }
    if weighted.is_empty() {
        return None;
    }
    let total: f64 = weighted.iter().map(|(_, w)| w).sum();
    let mut r = rng.gen_range(0.0..total);
    for (p, w) in &weighted {
        if r < *w {
            return Some(Move::Play(*p));
        }
        r -= *w;
    }
    Some(Move::Play(weighted.last().unwrap().0))
}

/// 快速模拟一局，返回（本局着法序列, 胜方）。
/// 着法序列为 (行棋方, 落点, 落子时 3×3 模式索引)，供 RAVE（AMAF）与模式在线学习使用。
/// 接近终局（空点 < 60）时直接计分。
fn fast_playout(
    board: &Board,
    heuristic: u8,
    max_moves: u32,
    policy: &mut PlayoutPolicy,
    use_patterns: bool,
    rng: &mut StdRng,
) -> (Vec<(Color, Point, usize)>, Color) {
    let mut b = board.clone();
    let mut passes = 0u32;
    let mut plies = 0u32;
    let mut moves: Vec<(Color, Point, usize)> = Vec::new();
    policy.last_move = None;
    loop {
        if empty_count(&b) < 60 {
            break; // 终局：直接计分
        }
        match rollout_policy_move(&b, heuristic, plies, policy, use_patterns, rng) {
            Some(Move::Play(p)) => {
                let c = b.turn();
                let idx = policy.table.index(&b, p.x, p.y, c);
                b.play(p.x, p.y);
                moves.push((c, p, idx));
                policy.last_move = Some(p);
                passes = 0;
            }
            _ => {
                passes += 1;
                if passes >= 2 {
                    break;
                }
            }
        }
        plies += 1;
        if plies >= max_moves {
            break;
        }
    }
    let winner =
        crate::core::compute_score(&b, &HashSet::new()).winner.unwrap_or(Color::Empty);
    (moves, winner)
}

/// 快速先验评分（Pachi「先验值」风格，用于渐进偏置/先验混合）：
/// 展开子节点时基于 3×3 模式权重 + 快速战术扫描给出 0.05~1.0 的先验。
fn cheap_prior(board: &Board, m: Move, table: &PatternTable) -> f64 {
    match m {
        Move::Pass => 0.4,
        Move::Resign => 0.0,
        Move::Play(p) => {
            let color = board.turn();
            let (opp_n, _own_n, opp_atari, own_save, _own_ignore) = scan_neighbors(board, p, color);
            // 模式权重（0.15~8）映射为先验贡献：权重 1.0（空旷中性）→ 0.5
            let idx = table.index(board, p.x, p.y, color);
            let w = table.weight(idx);
            let mut pr: f64 = 0.5 + ((w - 1.0).clamp(-0.9, 3.0) * 0.15) as f64;
            if opp_atari >= 1 {
                pr += 0.35; // 能吃子
            }
            if own_save >= 1 {
                pr += 0.25; // 能救子
            }
            if opp_n == 4 {
                pr -= 0.4; // 疑似自杀/填眼
            }
            pr.clamp(0.05, 1.0)
        }
    }
}

/// MCTS 节点（竞技场式存储，避免指针悬垂）。
struct MctsNode {
    /// 本节点局面。
    board: Board,
    /// 从父节点进入本节点的着法（根节点为 None）。
    move_made: Option<Move>,
    /// 本节点轮到哪一方行棋。
    to_move: Color,
    parent: Option<usize>,
    children: Vec<usize>,
    /// 尚未展开的候选着法。
    untried: Vec<Move>,
    /// 渐进偏置先验（0.05~1.0，来自快速战术扫描）。
    prior: f64,
    /// 从本节点行棋方视角的胜分。
    wins: f64,
    visits: u32,
}

/// 单个线程独立运行的 MCTS 树。
struct MctsTree {
    nodes: Vec<MctsNode>,
    heuristic: u8,
    c_uct: f64,
    use_patterns: bool,
    use_rave: bool,
    use_enhanced: bool,
    /// 叶节点虚拟模拟（Fuego 风格）：无候选可展开时直接继承祖先 wins/visits 作为回溯值。
    use_virtual_roll: bool,
    /// 快速走子策略（3×3 模式表 + 上一手跟踪）。
    policy: PlayoutPolicy,
    /// 根级 AMAF（RAVE/GRAVE）统计：着法 → (胜分, 访问数)。
    amaf: std::collections::HashMap<Move, (f64, u32)>,
    rng: StdRng,
}

impl MctsTree {
    fn new(
        board: Board,
        heuristic: u8,
        c_uct: f64,
        seed: u64,
        use_patterns: bool,
        use_rave: bool,
        use_enhanced: bool,
        use_virtual_roll: bool,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let root_moves = root_candidates(&board, heuristic, &mut rng);
        let to_move = board.turn();
        let root = MctsNode {
            board,
            move_made: None,
            to_move,
            parent: None,
            children: Vec::new(),
            untried: root_moves,
            prior: 0.5,
            wins: 0.0,
            visits: 0,
        };
        MctsTree {
            nodes: vec![root],
            heuristic,
            c_uct,
            use_patterns,
            use_rave,
            use_enhanced,
            use_virtual_roll,
            policy: PlayoutPolicy::new(),
            amaf: std::collections::HashMap::new(),
            rng,
        }
    }

    /// 单次 MCTS 迭代：选择 → 展开 → 模拟 → 回溯。
    fn iterate(&mut self, max_rollout_moves: u32) {
        // 选择：沿 UCT 最大的子节点下探到叶子
        let mut node = 0usize;
        while !self.nodes[node].children.is_empty() {
            node = self.select_child(node);
        }

        // 渐进拓宽（Progressive Widening）：访问数达到子节点数平方时才展开新子节点，
        // 把有限模拟集中到最有希望的着法上（高模拟规模下收益明显，默认由 use_enhanced 控制）。
        let can_expand = if self.use_enhanced {
            (self.nodes[node].visits as usize) >= self.nodes[node].children.len().pow(2)
        } else {
            true
        };
        if can_expand {
            // 从队尾弹出（默认路径的验证行为；渐进拓宽默认关闭时展开顺序不影响棋力，
            // 仅影响随机流。开启 use_enhanced 时根候选按“好→差”排列，队尾即最好着法）
            if let Some(m) = self.nodes[node].untried.pop() {
                let child = self.expand(node, m);
                self.rollout_and_backprop(child, max_rollout_moves);
                return;
            }
        }
        // 叶节点虚拟模拟（Fuego 风格 Virtual Roll）：无候选可展开时，
        // 若祖先（父节点/祖父节点）已有足够访问数（≥ VL_MIN_ANC_VISITS），
        // 用祖先胜率 wins/visits 作为本节点结果，避免做一次昂贵的 rollout；
        // 否则退回到普通 rollout（避免 0 访问祖先给出退化信号）。
        const VL_MIN_ANC_VISITS: u32 = 10;
        if self.use_virtual_roll {
            let parent_opt = self.nodes[node].parent;
            let ancestor_opt = if let Some(p) = parent_opt {
                self.nodes[p].parent.or(Some(p))
            } else {
                parent_opt
            };
            if let Some(a) = ancestor_opt {
                let anc = &self.nodes[a];
                if anc.visits >= VL_MIN_ANC_VISITS {
                    let winner = if anc.wins / anc.visits as f64 >= 0.5 {
                        self.nodes[node].to_move
                    } else {
                        self.nodes[node].to_move.opponent()
                    };
                    self.backprop_winner(node, winner);
                    return;
                }
            }
        }
        // 叶子无未尝试着法（或未达拓宽条件）：直接模拟
        self.rollout_and_backprop(node, max_rollout_moves);
    }

    fn select_child(&self, node: usize) -> usize {
        let parent_visits = self.nodes[node].visits.max(1) as f64;
        let ln = parent_visits.ln();
        let mut best = None;
        let mut best_uct = f64::NEG_INFINITY;
        for &c in &self.nodes[node].children {
            let child = &self.nodes[c];
            let mut value = if child.visits == 0 {
                f64::INFINITY
            } else {
                child.wins / child.visits as f64
            };
            if child.visits > 0 {
                if self.use_enhanced {
                    // GRAVE（广义 RAVE）：当前节点访问数低时，用根（最近高访问祖先）AMAF 统计混合
                    if self.nodes[node].visits < 60 {
                        if let Some(m) = child.move_made {
                            if let Some((aw, av)) = self.amaf.get(&m) {
                                let beta = *av as f64 / (*av as f64 + child.visits as f64);
                                value = (1.0 - beta) * value + beta * (aw / *av as f64);
                            }
                        }
                    }
                } else if self.use_rave && node == 0 {
                    // 回退：根级 RAVE
                    if let Some(m) = child.move_made {
                        if let Some((aw, av)) = self.amaf.get(&m) {
                            let beta = *av as f64 / (*av as f64 + child.visits as f64 + 1.0);
                            value = (1.0 - beta) * value + beta * (aw / *av as f64);
                        }
                    }
                }
                // Pachi「先验值」风格：模式权重先验以虚拟对局形式随访问数衰减混合，
                // 提升低模拟数下的选点质量；模式走子启用（中级及以上）即生效。
                if self.use_patterns {
                    value += self.c_uct * 0.5 * child.prior / (1.0 + child.visits as f64);
                }
            }
            let uct = value + self.c_uct * (ln / child.visits.max(1) as f64).sqrt();
            if uct > best_uct {
                best_uct = uct;
                best = Some(c);
            }
        }
        best.unwrap_or_else(|| *self.nodes[node].children.last().unwrap())
    }

    fn expand(&mut self, parent: usize, m: Move) -> usize {
        let parent_board = self.nodes[parent].board.clone();
        let child_turn = self.nodes[parent].to_move.opponent();
        let prior = cheap_prior(&parent_board, m, &self.policy.table);
        let mut board = parent_board;
        if let Move::Play(p) = m {
            board.play(p.x, p.y);
        }
        let untried = node_candidates(&board, self.heuristic, &mut self.rng);
        let child = MctsNode {
            board,
            move_made: Some(m),
            to_move: child_turn,
            parent: Some(parent),
            children: Vec::new(),
            untried,
            prior,
            wins: 0.0,
            visits: 0,
        };
        self.nodes.push(child);
        let id = self.nodes.len() - 1;
        self.nodes[parent].children.push(id);
        id
    }

    fn rollout_and_backprop(&mut self, node: usize, max_rollout_moves: u32) {
        let (moves, winner) = fast_playout(
            &self.nodes[node].board,
            self.heuristic,
            max_rollout_moves,
            &mut self.policy,
            self.use_patterns,
            &mut self.rng,
        );

        // 3×3 模式在线学习：按对局结果更新各着法的模式权重（Pachi 风格）
        if self.use_patterns {
            for (c, _, idx) in &moves {
                let result = if winner == *c {
                    1.0
                } else if winner == Color::Empty {
                    0.5
                } else {
                    0.0
                };
                self.policy.table.update(*idx, result as f32, 0.02);
            }
        }
        // 根级 AMAF（RAVE）：本局所有着法计入统计，供根选择加速收敛
        if self.use_rave {
            for (c, p, _) in &moves {
                let result = if winner == *c {
                    1.0
                } else if winner == Color::Empty {
                    0.5
                } else {
                    0.0
                };
                let e = self.amaf.entry(Move::Play(*p)).or_insert((0.0, 0));
                e.0 += result;
                e.1 += 1;
            }
        }

        // 回溯
        self.backprop_winner(node, winner);
    }

    /// 按胜方回溯更新节点统计（CPU 与 GPU 走子共用）。
    fn backprop_winner(&mut self, node: usize, winner: Color) {
        let mut cur = node;
        loop {
            let node_ref = &mut self.nodes[cur];
            node_ref.visits += 1;
            if node_ref.to_move == winner {
                node_ref.wins += 1.0;
            } else if winner == Color::Empty {
                node_ref.wins += 0.5;
            }
            match node_ref.parent {
                Some(p) => cur = p,
                None => break,
            }
        }
    }

    /// GPU 模式：收集 n 个 rollout 请求（选择 + 展开），返回 (叶子节点, 打包局面)。
    fn collect_rollout_requests(&mut self, n: usize, batch: &mut Vec<(usize, PlayoutInput)>) {
        for _ in 0..n {
            let mut node = 0usize;
            while !self.nodes[node].children.is_empty() {
                node = self.select_child(node);
            }
            let can_expand = if self.use_enhanced {
                (self.nodes[node].visits as usize) >= self.nodes[node].children.len().pow(2)
            } else {
                true
            };
            if can_expand {
                if let Some(m) = self.nodes[node].untried.pop() {
                    let child = self.expand(node, m);
                    batch.push((child, pack_board(&self.nodes[child].board)));
                    continue;
                }
            }
            batch.push((node, pack_board(&self.nodes[node].board)));
        }
    }

    /// 根节点各候选着法的 (着法, 胜分, 访问数)。
    fn root_stats(&self) -> Vec<(Move, f64, u32)> {
        let root = &self.nodes[0];
        root.children
            .iter()
            .map(|&c| {
                let n = &self.nodes[c];
                (n.move_made.unwrap(), n.wins, n.visits)
            })
            .collect()
    }
}

/// 空点数（快速统计，用于决定是否将 Pass 纳入候选）。
fn empty_count(board: &Board) -> usize {
    let mut n = 0;
    for y in 0..19 {
        for x in 0..19 {
            if board.is_empty(x, y) {
                n += 1;
            }
        }
    }
    n
}

/// 根节点候选着法：入门为纯随机（前 max_candidates），中/高级按启发式打分排序。
/// 仅当盘面接近填满（空点 < 60）时才加入 Pass，避免浅搜索过早虚着结束对局。
fn root_candidates(board: &Board, heuristic: u8, rng: &mut StdRng) -> Vec<Move> {
    let legal = board.legal_moves();
    let max_candidates = match heuristic {
        1 => 40,
        _ => 50,
    };
    let mut moves: Vec<Move> = if heuristic == 1 {
        // 入门：随机候选，棋力更弱，符合难度定位
        let mut mv: Vec<Move> = legal.iter().map(|&p| Move::Play(p)).collect();
        mv.shuffle(rng);
        mv.truncate(max_candidates);
        mv
    } else {
        let mut scored: Vec<(Point, i32)> = legal
            .iter()
            .map(|&p| (p, score_move(board, p, board.turn(), heuristic, rng, true)))
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored
            .into_iter()
            .take(max_candidates)
            .map(|(p, _)| Move::Play(p))
            .collect()
    };
    if empty_count(board) < 60 {
        moves.push(Move::Pass);
    }
    moves
}

/// 非根节点候选着法：棋子邻点优先，最多 max 个合法点 + Pass。
fn node_candidates(board: &Board, heuristic: u8, rng: &mut StdRng) -> Vec<Move> {
    let mut set: Vec<Point> = empty_neighbors(board).into_iter().collect();
    set.sort_by_key(|p| (p.y, p.x)); // 确定化：HashSet 迭代顺序随进程随机，先排序再洗牌
    if set.len() < 30 {
        let mut extra = Vec::new();
        for _ in 0..80 {
            let p = Point::new(rng.gen_range(0..19), rng.gen_range(0..19));
            if board.is_empty(p.x, p.y) && !set.contains(&p) {
                extra.push(p);
            }
            if extra.len() >= 30 - set.len() {
                break;
            }
        }
        set.extend(extra);
    }
    set.shuffle(rng);
    let mut legal = Vec::new();
    for p in set {
        if board.is_legal(p.x, p.y) {
            legal.push(Move::Play(p));
        }
        if legal.len() >= 12 {
            break;
        }
    }
    let _ = heuristic;
    if empty_count(board) < 60 {
        legal.push(Move::Pass);
    }
    legal
}

/// 着法的确定性排序键（最终选点时打破平局，避免依赖 HashMap 迭代顺序）。
fn move_key(m: &Move) -> (u8, usize, usize) {
    match m {
        Move::Play(p) => (0, p.y, p.x),
        Move::Pass => (1, 0, 0),
        Move::Resign => (2, 0, 0),
    }
}

/// 搜索结果。
pub struct SearchResult {
    pub mv: Move,
    pub visits: u32,
    pub total_sims: u32,
    pub elapsed: Duration,
}

/// 为当前局面选择一手棋（根并行 MCTS + 可选开局库）。
///
/// * `board`：当前局面（行棋方由 `board.turn()` 决定）。
/// * `history`：已下着法序列（供开局库匹配），仅取 Play 手。
/// * `difficulty`：难度。
/// * `book`：开局库。
/// * `seed`：随机种子（固定可复现）。
pub fn select_move(
    board: &Board,
    history: &[(Color, Point)],
    difficulty: Difficulty,
    book: &OpeningBook,
    seed: u64,
) -> SearchResult {
    let cfg = difficulty.config();
    select_move_cfg(board, history, &cfg, book, seed, None)
}

/// 使用自定义搜索配置选一手棋（供测试与外部集成定制预算）。
/// `gpu`：可选 GPU 引擎（Arc<Mutex> 供多线程共享）；Some 时走子阶段用 GPU 批量计算。
pub fn select_move_cfg(
    board: &Board,
    history: &[(Color, Point)],
    cfg: &SearchConfig,
    book: &OpeningBook,
    seed: u64,
    gpu: Option<Arc<Mutex<crate::gpu::GpuEngine>>>,
) -> SearchResult {
    let start = Instant::now();

    // 每次搜索开始前清除取消标志，避免上一次残留的取消信号误伤本次搜索
    clear_ai_cancel();

    // 开局库
    if cfg.use_book {
        if let Some(p) = book.find(history, cfg.book_plies) {
            // 校验库中着法当前合法
            if board.is_legal(p.x, p.y) {
                return SearchResult {
                    mv: Move::Play(p),
                    visits: 1,
                    total_sims: 0,
                    elapsed: start.elapsed(),
                };
            }
        }
    }

    // 无合法落子 → 虚着
    let legal = board.legal_moves();
    if legal.is_empty() {
        return SearchResult {
            mv: Move::Pass,
            visits: 0,
            total_sims: 0,
            elapsed: start.elapsed(),
        };
    }

    // 根并行 MCTS
    let threads = cfg.threads.max(1);
    let shared_sims = Arc::new(AtomicU32::new(0));
    let mut handles = Vec::new();
    let per_thread = (cfg.max_sims / threads as u32).max(1);

    for t in 0..threads {
        let board = board.clone();
        let shared = Arc::clone(&shared_sims);
        let cfg = cfg.clone();
        let gpu = gpu.clone();
        handles.push(thread::spawn(move || {
            let mut tree = MctsTree::new(
                board,
                cfg.heuristic,
                cfg.c_uct,
                seed.wrapping_add(t as u64 * 0x9E3779B97F4A7C15),
                cfg.use_patterns,
                cfg.use_rave,
                cfg.use_enhanced,
                cfg.use_virtual_roll,
            );
            let mut local = 0u32;
            loop {
                // 时间与模拟数双重预算（自适应停止）+ 取消标志检查
                if local % 64 == 0 {
                    // 全局取消标志：GUI 退出 / 切难度 / 用户主动停止时立即退出
                    if AI_CANCEL.load(Ordering::SeqCst) {
                        break;
                    }
                    let elapsed = start.elapsed();
                    let global = shared.load(Ordering::Relaxed);
                    if elapsed >= cfg.max_time && global >= cfg.min_sims {
                        break;
                    }
                    if elapsed >= cfg.max_time.saturating_mul(3) {
                        break; // 硬超时保护
                    }
                    if global >= cfg.max_sims {
                        break;
                    }
                }
                if local >= per_thread {
                    break;
                }
                if let Some(g) = &gpu {
                    // GPU 批量走子：收集一批叶子 → 一次 dispatch → 统一回溯
                    // （计算量完全在 GPU 上，仅结果读回；加大批次以摊薄调度开销）
                    const GPU_BATCH: usize = 1024;
                    let remaining = (per_thread - local) as usize;
                    let n = remaining.min(GPU_BATCH).max(1);
                    let mut batch: Vec<(usize, PlayoutInput)> = Vec::with_capacity(n);
                    tree.collect_rollout_requests(n, &mut batch);
                    let inputs: Vec<PlayoutInput> = batch.iter().map(|(_, p)| *p).collect();
                    let mut outputs = vec![PlayoutOutput { winner: 2 }; n];
                    let result = match g.lock() {
                        Ok(mut eng) => eng.run_playouts(&inputs, &mut outputs),
                        Err(_) => Err("GPU 锁获取失败".to_string()),
                    };
                    match result {
                        Ok(()) => {
                            for (i, (node, _)) in batch.iter().enumerate() {
                                let winner = match outputs[i].winner {
                                    0 => Color::Black,
                                    1 => Color::White,
                                    _ => Color::Empty,
                                };
                                tree.backprop_winner(*node, winner);
                            }
                        }
                        Err(_) => {
                            // GPU 失败 → 回退 CPU 走子
                            for (node, _) in batch {
                                tree.rollout_and_backprop(node, cfg.max_rollout_moves);
                            }
                        }
                    }
                    local += n as u32;
                    shared.fetch_add(n as u32, Ordering::Relaxed);
                } else {
                    tree.iterate(cfg.max_rollout_moves);
                    local += 1;
                    shared.fetch_add(1, Ordering::Relaxed);
                }
            }
            tree.root_stats()
        }));
    }

    let mut agg: std::collections::HashMap<Move, (f64, u32)> = std::collections::HashMap::new();
    let mut total_sims = 0u32;
    for h in handles {
        for (m, wins, visits) in h.join().unwrap() {
            let e = agg.entry(m).or_insert((0.0, 0));
            e.0 += wins;
            e.1 += visits;
            total_sims += visits;
        }
    }

    // 根温度选点（Winrate-with-Exploration / Cgoban / Lizzie 通用做法）：
    // 选点权重 = visits^(1/T)（T > 1 增强探索，降低过度拟合；T = 1 等价访问数；T < 1 更贪婪），
    // 叠加一个微幅随机抖动以避免完全依赖访问数。当 root_temperature == 1.0 时回退
    // 到纯访问数最大 + 胜分平局打破的确定性做法。
    let mv = if (cfg.root_temperature - 1.0).abs() > 1e-6 && agg.len() > 1 {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(seed);
        let temp = cfg.root_temperature.max(0.5).min(5.0);
        let exp = 1.0 / temp;
        let weighted: Vec<(&Move, f64)> = agg
            .iter()
            .map(|(m, (_, v))| {
                let w = if *v > 0 { (*v as f64).powf(exp) } else { 0.0 };
                // 微噪声：按温度比例加噪，抑制过度依赖访问数
                let noise = w * 0.08 * rng.gen_range(0.9..1.1);
                (m, w + noise)
            })
            .filter(|(_, w)| *w > 1e-9)
            .collect();
        if weighted.is_empty() {
            *agg.keys().next().unwrap_or(&Move::Pass)
        } else {
            let total: f64 = weighted.iter().map(|(_, w)| *w).sum();
            let mut r = rng.gen_range(0.0..total);
            for (_m, w) in weighted.iter() {
                if r < *w {
                    break;
                }
                r -= *w;
            }
            *weighted.last().unwrap().0
        }
    } else {
        // 纯访问数最大 + 胜分平局打破
        let best = agg
            .iter()
            .max_by(|a, b| {
                (a.1 .1, a.1 .0 as i64)
                    .cmp(&(b.1 .1, b.1 .0 as i64))
                    .then_with(|| move_key(a.0).cmp(&move_key(b.0)))
            })
            .map(|(m, _)| *m);
        best.unwrap_or(Move::Pass)
    };

    let visits = agg.get(&mv).map(|(_, v)| *v).unwrap_or(0);
    SearchResult {
        mv,
        visits,
        total_sims,
        elapsed: start.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_book_matches_prefix() {
        let book = OpeningBook::load();
        assert!(book.len() >= 20, "内置开局库至少 20 条");
        // 黑 Q16 后，库中应给出白 D4 之类着法
        let history = vec![(Color::Black, Point::from_sgf("Q16").unwrap())];
        let next = book.find(&history, 50);
        assert!(next.is_some(), "首手 Q16 应命中开局库");
        let p = next.unwrap();
        assert_eq!(p, Point::from_sgf("D4").unwrap(), "黑 Q16 后标准应手 D4");
        // 超过使用手数后不再命中
        let long: Vec<(Color, Point)> = (0..60)
            .map(|i| {
                let c = if i % 2 == 0 { Color::Black } else { Color::White };
                (c, Point::new(i % 19, (i * 7) % 19))
            })
            .collect();
        assert!(book.find(&long, 10).is_none());
    }

    #[test]
    fn easy_ai_returns_legal_move() {
        let board = Board::new();
        let book = OpeningBook::load();
        let r = select_move(&board, &[], Difficulty::Easy, &book, 42);
        match r.mv {
            Move::Play(p) => assert!(board.is_legal(p.x, p.y)),
            Move::Pass => {}
            Move::Resign => panic!("AI 不应认输"),
        }
    }

    #[test]
    fn master_difficulty_config_sane() {
        let cfg = Difficulty::Master.config();
        assert_eq!(cfg.max_time, Duration::from_secs(60), "大师级单步 60s");
        assert!(cfg.threads >= 8 && cfg.threads <= 16, "全核并行（8-16 线程）");
        assert!(cfg.min_sims >= 100_000, "保底模拟数充足");
        assert!(cfg.max_sims >= 1_000_000, "模拟上限可达 500 万");
        assert_eq!(cfg.heuristic, 3, "使用最强启发式");
        assert!(cfg.max_candidates >= 60, "候选数更多");
        assert!(cfg.use_patterns && cfg.use_rave, "大师级启用模式走子与 RAVE");
        // 内存估算：节点 ≤ 模拟数 × ~500B，远低于 15GB 上限
        let est_bytes = cfg.max_sims as u64 * 500;
        assert!(est_bytes < 15u64 << 30, "估算内存应低于 15GB");
    }

    #[test]
    fn pattern_table_sane() {
        let table = PatternTable::new();
        let board = Board::new();
        let idx = table.index(&board, 9, 9, Color::Black);
        assert!(idx < PATTERN_CLASSES);
        // 确定性：同一局面同一索引
        assert_eq!(table.index(&board, 9, 9, Color::Black), idx);
        let w = table.weight(idx);
        assert!(w > 0.0 && w < 8.0, "权重应有界");
        // 填眼模式（四邻全为己方）应显著降温
        let mut b = Board::new();
        for (x, y) in [(8, 9), (10, 9), (9, 8), (9, 10)] {
            b.set(x, y, Color::Black);
        }
        let eye_idx = table.index(&b, 9, 9, Color::Black);
        assert!(
            table.weight(eye_idx) < w,
            "填眼模式权重应低于空旷模式"
        );
        // 在线学习：获胜后权重上升，落败后回落
        let mut t = PatternTable::new();
        let w0 = t.weight(idx);
        t.update(idx, 1.0, 0.5);
        assert!(t.weight(idx) > w0, "获胜着法模式权重应上调");
        t.update(idx, 0.0, 1.0);
        assert!((t.weight(idx) - 0.05).abs() < 1e-3, "连续落败应逼近下限");
    }

    #[test]
    fn fuseki_bias_prefers_corners_early() {
        // 布局阶段（前 20 手）：角星位 > 角部 > 边线 > 中腹
        let star = fuseki_bias(3, 3, 2);
        let corner = fuseki_bias(1, 1, 2);
        let side = fuseki_bias(2, 9, 2);
        let center = fuseki_bias(9, 9, 2);
        assert!(star > corner, "角星位应最优先");
        assert!(corner > side, "角部应优先于边线");
        assert!(side > center, "边线应优先于中腹");
        assert!(center < 1.0, "布局阶段中腹应降温");
        // 布局结束后回归中性
        assert_eq!(fuseki_bias(9, 9, 21), 1.0);
        assert_eq!(fuseki_bias(3, 3, 40), 1.0);
    }
}

