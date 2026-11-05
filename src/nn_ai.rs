//! 神经网络 AI：把训练好的 policy+value 双头网络接进对弈，提供三档难度。
//!
//! 三档设计思路 —— 用「搜索深度 + 采样温度」拉开棋力梯度：
//!
//! | 档位 | 依据 | 模拟数 | 选点方式 | 单步耗时 |
//! |------|------|--------|----------|----------|
//! | 低   | 纯策略 | 0  | top-5 高温采样，会选到次优甚至第三好的点 | ~10ms |
//! | 中   | 纯策略 | 0  | top-12 中温采样，多数情况贴网络首选 | ~10ms |
//! | 高   | MCTS   | 160 | PUCT 搜索 + 价值头估值，低温柔性取最优 | ~1.4s |
//!
//! 低/中档刻意保留随机性 —— 只有会犯错，才谈得上「难度」。

use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::ai::SearchResult;
use crate::core::{Board, Color, Move, Point};
use crate::nn::{self, HISTORY, PASS_IDX, POLICY_SIZE};

/// AI 引擎选择：经典（MCTS + 启发式）或神经网络（训练好的双头网络）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AiEngine {
    /// 经典引擎：MCTS + 启发式快速走子 + 开局库（四档）。
    Classic,
    /// 神经网络引擎：policy+value 双头网络（低／中／高三档）。
    Neural,
}

impl AiEngine {
    pub fn label(&self) -> &'static str {
        match self {
            AiEngine::Classic => "经典",
            AiEngine::Neural => "神经网络",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            AiEngine::Classic => "传统蒙特卡洛树搜索 + 启发式走子，棋力稳定、不依赖模型文件",
            AiEngine::Neural => "用近千万局棋谱训练的双头网络（策略+价值），落子更像人类",
        }
    }

    /// 从命令行字符串解析。
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "classic" | "classical" | "mcts" | "经典" | "old" => Some(AiEngine::Classic),
            "nn" | "neural" | "net" | "神经网络" | "ai" => Some(AiEngine::Neural),
            _ => None,
        }
    }
}

/// 神经网络三档难度。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NnDifficulty {
    /// 低：纯策略 + 高温采样，明显会失误。
    Low,
    /// 中：纯策略 + 中温采样，偶有失误。
    Medium,
    /// 高：MCTS 搜索 + 价值头估值。
    High,
}

impl NnDifficulty {
    pub fn label(&self) -> &'static str {
        match self {
            NnDifficulty::Low => "低",
            NnDifficulty::Medium => "中",
            NnDifficulty::High => "高",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            NnDifficulty::Low => "神经网络快速出招，随机性较大，适合初学者熟悉规则",
            NnDifficulty::Medium => "神经网络直接选点，多数时候走网络最看好的位置",
            NnDifficulty::High => "神经网络 + 树搜索（160 次模拟），会算清局部变化",
        }
    }

    /// 从命令行/配置字符串解析（支持中英文与数字）。
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" | "低" | "1" | "easy" => Some(NnDifficulty::Low),
            "medium" | "中" | "2" | "mid" => Some(NnDifficulty::Medium),
            "high" | "高" | "3" | "hard" => Some(NnDifficulty::High),
            _ => None,
        }
    }

    pub fn all() -> [NnDifficulty; 3] {
        [NnDifficulty::Low, NnDifficulty::Medium, NnDifficulty::High]
    }

    pub fn config(&self) -> NnConfig {
        match self {
            NnDifficulty::Low => NnConfig {
                sims: 0,
                temperature: 2.0,
                top_k: 5,
                c_puct: 1.5,
                max_time: Duration::from_millis(800),
            },
            NnDifficulty::Medium => NnConfig {
                sims: 0,
                temperature: 0.9,
                top_k: 12,
                c_puct: 1.5,
                max_time: Duration::from_millis(800),
            },
            NnDifficulty::High => NnConfig {
                sims: 160,
                temperature: 0.25,
                top_k: 3,
                c_puct: 1.6,
                max_time: Duration::from_secs(5),
            },
        }
    }
}

/// 单档具体参数。
#[derive(Clone, Copy, Debug)]
pub struct NnConfig {
    /// MCTS 模拟次数，0 表示纯策略（不做搜索）。
    pub sims: u32,
    /// 选点温度：越大越随机，越小越贴近网络首选。
    pub temperature: f32,
    /// 只在先验最高的前 k 个着法里挑选（克制盲目乱走）。
    pub top_k: usize,
    /// PUCT 探索系数。
    pub c_puct: f64,
    /// 单步时间上限（防止极端局面卡死）。
    pub max_time: Duration,
}

/// 三档统一入口。`snaps[0]` 必须是当前局面，其余为历史局面（按时间倒序）。
pub fn select_move_nn(snaps: &[Board], diff: NnDifficulty, seed: u64) -> Option<SearchResult> {
    let start = Instant::now();
    let board = snaps.first()?;
    let to_move = board.turn();

    if !nn::is_ready() {
        return None;
    }

    let legal: Vec<Point> = board.legal_moves();
    if legal.is_empty() {
        return Some(SearchResult {
            mv: Move::Pass,
            visits: 0,
            total_sims: 0,
            elapsed: start.elapsed(),
        });
    }

    let cfg = diff.config();
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5DEE_CE66);

    let mv = if cfg.sims == 0 {
        // 纯策略：一次前向，按温度在 top-k 内采样
        let ev = nn::evaluate(&as_refs(snaps), to_move)?;
        sample_from_policy(&ev.policy, &legal, &cfg, &mut rng)
    } else {
        // MCTS + 网络先验 / 价值头
        let mut tree = SearchTree::new(snaps.to_vec(), cfg.c_puct);
        let deadline = start + cfg.max_time;
        for _ in 0..cfg.sims {
            if Instant::now() >= deadline {
                break;
            }
            tree.iterate();
        }
        tree.best_move(&legal, &cfg, &mut rng)
    };

    Some(SearchResult {
        mv,
        visits: cfg.sims,
        total_sims: cfg.sims,
        elapsed: start.elapsed(),
    })
}

fn as_refs(v: &[Board]) -> Vec<&Board> {
    v.iter().collect()
}

/// 把 362 维 policy 转成「着法 → 概率」，按温度/top-k 采样。
fn sample_from_policy(
    policy: &[f32; POLICY_SIZE],
    legal: &[Point],
    cfg: &NnConfig,
    rng: &mut StdRng,
) -> Move {
    let probs = nn::softmax(policy);

    let mut cands: Vec<(Point, f32)> = legal
        .iter()
        .map(|&p| (p, probs[p.y * nn::BOARD + p.x]))
        .filter(|(_, pr)| *pr > 0.0)
        .collect();
    cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    cands.truncate(cfg.top_k.max(1));

    if cands.is_empty() {
        return Move::Play(legal[rng.gen_range(0..legal.len())]);
    }

    // 温度调节：w = p^(1/T)
    let inv_t = 1.0 / cfg.temperature.max(1e-3) as f64;
    let weights: Vec<f64> = cands.iter().map(|(_, p)| (*p as f64).powf(inv_t)).collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return Move::Play(cands[0].0);
    }
    let mut r = rng.gen::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return Move::Play(cands[i].0);
        }
    }
    Move::Play(cands[cands.len() - 1].0)
}

// ---------------------------------------------------------------- MCTS

struct Node {
    board: Board,
    parent: Option<usize>,
    /// 从父节点进入本节点所用的着法（根为 None）。
    move_made: Option<Move>,
    children: Vec<usize>,
    /// 尚未展开的着法及其先验（按先验升序，`pop` 取出最后一个 = 最高先验）。
    untried: Vec<(Move, f32)>,
    /// 本节点被选中时的先验概率。
    prior: f32,
    n: u32,
    /// 累积价值，**本节点行棋方**视角。
    w: f64,
    to_move: Color,
    /// 本节点的网络估值缓存，完全展开后直接复用。
    value: f32,
}

struct SearchTree {
    nodes: Vec<Node>,
    c_puct: f64,
    /// 根节点及其之前的历史局面（[0] = 根局面），用于补齐特征的历史平面。
    root_snaps: Vec<Board>,
}

impl SearchTree {
    fn new(snaps: Vec<Board>, c_puct: f64) -> Self {
        let board = snaps[0].clone();
        let to_move = board.turn();
        let ev = nn::evaluate(&as_refs(&snaps), to_move);

        let untried = build_untried(&board, ev.as_ref());
        let root = Node {
            board,
            parent: None,
            move_made: None,
            children: Vec::new(),
            untried,
            prior: 1.0,
            n: 0,
            w: 0.0,
            to_move,
            value: ev.map(|e| e.value).unwrap_or(0.0),
        };
        SearchTree {
            nodes: vec![root],
            c_puct,
            root_snaps: snaps,
        }
    }

    fn select_child(&self, idx: usize) -> usize {
        let parent_n = self.nodes[idx].n as f64;
        let mut best = self.nodes[idx].children[0];
        let mut best_score = f64::NEG_INFINITY;
        for &c in &self.nodes[idx].children {
            let ch = &self.nodes[c];
            // 子节点累积的是「子节点行棋方」视角价值，换回父视角要取负
            let q = if ch.n > 0 { -ch.w / ch.n as f64 } else { 0.0 };
            let u = self.c_puct * ch.prior as f64 * parent_n.sqrt() / (1.0 + ch.n as f64);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = c;
            }
        }
        best
    }

    /// 构造「把 b 作为新叶子」时的 4 个历史快照（[0]=b）。
    fn snaps_for(&self, parent: usize, b: &Board) -> Vec<Board> {
        let mut out: Vec<Board> = Vec::with_capacity(HISTORY);
        out.push(b.clone());
        let mut cur = Some(parent);
        while let Some(i) = cur {
            if out.len() >= HISTORY {
                break;
            }
            out.push(self.nodes[i].board.clone());
            cur = self.nodes[i].parent;
        }
        // 还不够就用根之前的真实历史补齐（root_snaps[0] 已用过，从 1 开始）
        for b2 in self.root_snaps.iter().skip(1) {
            if out.len() >= HISTORY {
                break;
            }
            out.push(b2.clone());
        }
        out
    }

    fn iterate(&mut self) {
        // 1) 选择
        let mut node = 0usize;
        while !self.nodes[node].children.is_empty() {
            node = self.select_child(node);
        }

        // 2) 展开（若有未展开着法）
        if let Some((mv, prior)) = self.nodes[node].untried.pop() {
            let mut b = self.nodes[node].board.clone();
            let ok = match mv {
                Move::Play(p) => b.play(p.x, p.y),
                Move::Pass => {
                    b.clear_ko();
                    true
                }
                Move::Resign => false,
            };
            if !ok {
                return;
            }
            // 虚着不换手（沿用现有 Game 语义），避免与对局层不一致
            let to_move = b.turn();
            let snaps = self.snaps_for(node, &b);
            let ev = nn::evaluate(&as_refs(&snaps), to_move);
            let value = ev.as_ref().map(|e| e.value).unwrap_or(0.0);
            let untried = build_untried(&b, ev.as_ref());

            let leaf = self.nodes.len();
            self.nodes.push(Node {
                board: b,
                parent: Some(node),
                move_made: Some(mv),
                children: Vec::new(),
                untried,
                prior,
                n: 0,
                w: 0.0,
                to_move,
                value,
            });
            self.nodes[node].children.push(leaf);
            self.backprop(leaf, value);
        } else {
            // 3) 已完全展开：直接用缓存估值回溯
            let v = self.nodes[node].value;
            self.backprop(node, v);
        }
    }

    fn backprop(&mut self, leaf: usize, leaf_value: f32) {
        let mut v = leaf_value as f64;
        let mut cur = Some(leaf);
        while let Some(i) = cur {
            self.nodes[i].n += 1;
            self.nodes[i].w += v;
            v = -v; // 上一层行棋方相反
            cur = self.nodes[i].parent;
        }
    }

    fn best_move(&self, legal: &[Point], cfg: &NnConfig, rng: &mut StdRng) -> Move {
        let root = &self.nodes[0];
        let mut cands: Vec<(Move, u32)> = root
            .children
            .iter()
            .filter_map(|&c| {
                let n = self.nodes[c].n;
                if n == 0 {
                    None
                } else {
                    self.nodes[c].move_made.map(|m| (m, n))
                }
            })
            .collect();

        if cands.is_empty() {
            // 搜索没能展开任何节点 → 退回纯策略单次前向
            if let Some(ev) = nn::evaluate(&as_refs(&self.root_snaps), root.to_move) {
                return sample_from_policy(&ev.policy, legal, cfg, rng);
            }
            return if legal.is_empty() {
                Move::Pass
            } else {
                Move::Play(legal[0])
            };
        }

        cands.sort_by(|a, b| b.1.cmp(&a.1));
        let max_n = cands[0].1;

        // 只保留访问数接近最优的候选，再按温度加权（避免纯 argmax 的僵化）
        let top: Vec<(Move, f64)> = cands
            .iter()
            .filter(|(_, n)| (*n as f64) >= max_n as f64 * 0.35)
            .map(|(m, n)| (*m, (*n as f64).powf(1.0 / cfg.temperature.max(1e-3) as f64)))
            .collect();
        let total: f64 = top.iter().map(|(_, w)| *w).sum();
        if total <= 0.0 || !total.is_finite() {
            return cands[0].0;
        }
        let mut r = rng.gen::<f64>() * total;
        for (m, w) in &top {
            r -= w;
            if r <= 0.0 {
                return *m;
            }
        }
        top[top.len() - 1].0
    }
}

/// 由网络 policy 生成「未展开着法 + 先验」，按先验升序（`pop` 取最高先验）。
fn build_untried(board: &Board, ev: Option<&nn::NnEval>) -> Vec<(Move, f32)> {
    let legal = board.legal_moves();
    let mut v: Vec<(Move, f32)> = match ev {
        Some(e) => {
            let probs = nn::softmax(&e.policy);
            legal
                .iter()
                .map(|&p| (Move::Play(p), probs[p.y * nn::BOARD + p.x]))
                .chain(std::iter::once((Move::Pass, probs[PASS_IDX])))
                .collect()
        }
        None => legal.iter().map(|&p| (Move::Play(p), 1.0)).collect(),
    };
    v.retain(|(_, p)| *p > 1e-4);
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}
