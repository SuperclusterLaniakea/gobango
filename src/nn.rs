//! 神经网络推理模块：加载训练好的 policy+value 双头网络，供对弈使用。
//!
//! 网络结构（见训练工程 go2go/go-ai/model.py）：
//!   输入 9×19×19 → stem Conv3x3 → 6 × ResBlock(128ch) → 双头
//!   policy: 362 类（361 个交叉点 + PASS）
//!   value : tanh 输出 [-1,1]，表示**当前行棋方**的胜率倾向（正 = 有利）
//!
//! 输入特征必须与训练时逐位一致（否则权重完全失效）：
//!   平面 0..3  = 当前 / 1 手前 / 2 手前 / 3 手前的「行棋方棋子」
//!   平面 4..7  = 同一时刻的「对手棋子」
//!   平面 8     = 常数平面（轮到白方 = 1，黑方 = 0）
//! 其中快照不足 4 个时补零，与 Python 侧 `np.zeros((HISTORY,...))` 行为一致。

use std::io::Cursor;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use tract_onnx::prelude::*;

use crate::core::{Board, Color};

/// 棋盘尺寸（训练与推理都固定 19 路）。
pub const BOARD: usize = 19;
/// 历史快照数。
pub const HISTORY: usize = 4;
/// 输入平面数 = 己方 4 + 对手 4 + 常数 1。
pub const PLANES: usize = 2 * HISTORY + 1;
/// 交叉点数量。
pub const POINTS: usize = BOARD * BOARD;
/// PASS 在 policy 输出中的下标。
pub const PASS_IDX: usize = POINTS;
/// policy 输出维度。
pub const POLICY_SIZE: usize = POINTS + 1;

/// 编译期内嵌的模型权重，保证单文件部署。
static EMBEDDED_MODEL: &[u8] = include_bytes!("../assets/policy_value.onnx");

type Plan = SimplePlan<TypedFact, Box<dyn TypedOp>, TypedModel>;

/// 一次评估的结果。
#[derive(Clone)]
pub struct NnEval {
    /// 362 维策略概率（已做 softmax 前的 logits 归一化由调用方决定，这里直接给概率）。
    pub policy: [f32; POLICY_SIZE],
    /// 价值，[-1,1]，**当前行棋方**视角。
    pub value: f32,
}

struct Engine {
    plan: Plan,
}

/// 全局引擎，只加载一次。`None` 表示加载失败（功能降级，不影响经典引擎）。
static ENGINE: OnceLock<Option<Mutex<Engine>>> = OnceLock::new();

fn load_from(path: &Path) -> TractResult<Plan> {
    let bytes = std::fs::read(path)?;
    build_plan(&bytes)
}

fn build_plan(bytes: &[u8]) -> TractResult<Plan> {
    tract_onnx::onnx()
        .model_for_read(&mut Cursor::new(bytes))?
        .into_optimized()?
        .into_runnable()
}

fn engine() -> Option<&'static Mutex<Engine>> {
    ENGINE
        .get_or_init(|| match build_plan(EMBEDDED_MODEL) {
            Ok(plan) => Some(Mutex::new(Engine { plan })),
            Err(e) => {
                eprintln!("[nn] 内嵌模型加载失败：{e}");
                None
            }
        })
        .as_ref()
}

/// 初始化：优先读取外部模型文件（便于替换权重），失败则回退到内嵌模型。
///
/// 返回 `true` 表示神经网络可用。
pub fn init(external: Option<&Path>) -> bool {
    if let Some(p) = external {
        if p.is_file() {
            match load_from(p) {
                Ok(plan) => {
                    // 若已初始化过内嵌版本，这里用外部权重覆盖（OnceLock 只能设置一次，
                    // 故用 Mutex 内部替换来支持热替换）
                    let holder = ENGINE.get_or_init(|| Some(Mutex::new(Engine { plan })));
                    if let Some(m) = holder {
                        if let Ok(mut g) = m.lock() {
                            // 重新加载一次，避免所有权问题
                            if let Ok(p2) = load_from(p) {
                                g.plan = p2;
                            }
                        }
                        eprintln!("[nn] 已加载外部模型：{}", p.display());
                    }
                    return true;
                }
                Err(e) => eprintln!("[nn] 外部模型加载失败（回退内嵌）：{e}"),
            }
        }
    }
    engine().is_some()
}

/// 神经网络是否可用。
pub fn is_ready() -> bool {
    engine().is_some()
}

/// 构造 9 平面输入特征。
///
/// `snaps[0]` 必须是**当前局面**，`snaps[1]` 是上一手落子后的局面，依此类推。
/// 行棋方由 `to_move` 显式指定（与 `board.turn()` 解耦，便于搜树时复用）。
fn build_input(snaps: &[&Board], to_move: Color) -> tract_ndarray::Array4<f32> {
    let mut arr = tract_ndarray::Array4::<f32>::zeros((1, PLANES, BOARD, BOARD));
    let opp = to_move.opponent();

    for (i, b) in snaps.iter().take(HISTORY).enumerate() {
        for y in 0..BOARD {
            for x in 0..BOARD {
                let c = b.at(x, y);
                // grid[row=y][col=x]，与 Python 的 arrs[i][r][c] 对齐
                arr[[0, i, y, x]] = if c == to_move { 1.0 } else { 0.0 };
                arr[[0, HISTORY + i, y, x]] = if c == opp { 1.0 } else { 0.0 };
            }
        }
    }
    // 常数平面：轮到白方为 1。这是区分「这一手轮到谁」的绝对颜色信息，
    // 缺了它网络无法表达贴目带来的先后手差异（训练时已验证）。
    let constv = if to_move == Color::White { 1.0 } else { 0.0 };
    for y in 0..BOARD {
        for x in 0..BOARD {
            arr[[0, 2 * HISTORY, y, x]] = constv;
        }
    }
    arr
}

/// 对给定局面（含历史快照）做一次推理。
pub fn evaluate(snaps: &[&Board], to_move: Color) -> Option<NnEval> {
    let holder = engine()?;
    let input = build_input(snaps, to_move);

    let mut guard = holder.lock().ok()?;
    let outputs = guard
        .plan
        .run(tvec!(input.into_tensor().into()))
        .ok()?;

    let policy_view = outputs.first()?.to_array_view::<f32>().ok()?;
    let mut policy = [0.0f32; POLICY_SIZE];
    for (i, v) in policy_view.iter().take(POLICY_SIZE).enumerate() {
        policy[i] = *v;
    }

    let value = outputs
        .get(1)
        .and_then(|t| t.to_array_view::<f32>().ok())
        .and_then(|v| v.iter().next().copied())
        .unwrap_or(0.0);

    Some(NnEval { policy, value })
}

/// 对 policy logits 做 softmax，返回概率。
pub fn softmax(logits: &[f32; POLICY_SIZE]) -> [f32; POLICY_SIZE] {
    let mut out = [0.0f32; POLICY_SIZE];
    let maxv = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for (i, &v) in logits.iter().enumerate() {
        let e = ((v - maxv) as f64).exp();
        out[i] = e as f32;
        sum += e;
    }
    if sum > 0.0 {
        for v in out.iter_mut() {
            *v = (*v as f64 / sum) as f32;
        }
    }
    out
}

/// 便捷封装：对单个棋盘（无历史）评估。
pub fn eval_board(board: &Board) -> Option<NnEval> {
    evaluate(&[board], board.turn())
}
