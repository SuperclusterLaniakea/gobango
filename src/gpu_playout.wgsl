// GobanGo GPU 批量走子着色器（计算着色器，每个线程独立完成一局随机走子）
//
// 每个位置占用 324 个 u32：
//   [0..23]    棋盘：每格 2 位（0=空 1=黑 2=白），每 u32 存 16 格
//   [24]       低 2 位=行棋方（1=黑 2=白）；bit2..10=劫点索引（0..360，511=无）
//   [25..324]  该线程的栈空间（洪泛/候选用，各流程串行复用）
//
// 走子策略为局部启发式（吃子代理/接触/禁自杀/防填眼 + 噪声），
// 规则正确性由落子后的提子与自杀回滚保证；终局按中国规则（子数+围空）判定胜负。

const TOTAL: u32 = 361u;
const STACK_BASE: u32 = 25u;
const STACK_MAX: u32 = 280u;
const NO_KO: u32 = 511u;
const MAX_PLIES: u32 = 200u;
const CAND_MAX: u32 = 24u;

struct PlayoutIn {
    data: array<u32, 324>,
}

struct PlayoutOut {
    winner: u32,
}

struct Params {
    seed: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read_write> inputs: array<PlayoutIn>;
@group(0) @binding(1) var<storage, read_write> outputs: array<PlayoutOut>;
@group(0) @binding(2) var<uniform> params: Params;

var<private> rng_state: u32 = 0u;

fn rng_next() -> u32 {
    var x = rng_state;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    rng_state = x;
    return x;
}

fn rand_range(n: u32) -> u32 {
    return rng_next() % n;
}

fn cell_at(id: u32, idx: u32) -> u32 {
    let w = idx / 16u;
    let s = (idx % 16u) * 2u;
    return (inputs[id].data[w] >> s) & 3u;
}

fn set_cell(id: u32, idx: u32, val: u32) {
    let w = idx / 16u;
    let s = (idx % 16u) * 2u;
    let mask = 3u << s;
    inputs[id].data[w] = (inputs[id].data[w] & ~mask) | (val << s);
}

fn pt_idx(x: u32, y: u32) -> u32 {
    return y * 19u + x;
}

// ---- 洪泛工具（使用每线程栈区，标记值 3 表示“已访问”同色链）----

// 将包含 start 的 color 链标记为 3，返回气数（调用方需自行恢复/转换）
fn flood_mark(id: u32, start: u32, color: u32) -> u32 {
    var sp = 0u;
    inputs[id].data[STACK_BASE + sp] = start;
    sp += 1u;
    set_cell(id, start, 3u);
    var libs = 0u;
    var guard = 0u;
    while sp > 0u && guard < 4000u {
        guard += 1u;
        sp -= 1u;
        let cur = inputs[id].data[STACK_BASE + sp];
        let cx = cur % 19u;
        let cy = cur / 19u;
        if cx >= 1u {
            let nb = pt_idx(cx - 1u, cy);
            let c = cell_at(id, nb);
            if c == 0u {
                libs += 1u;
            } else if c == color {
                set_cell(id, nb, 3u);
                inputs[id].data[STACK_BASE + sp] = nb;
                sp += 1u;
            }
        }
        if cx <= 17u {
            let nb = pt_idx(cx + 1u, cy);
            let c = cell_at(id, nb);
            if c == 0u {
                libs += 1u;
            } else if c == color {
                set_cell(id, nb, 3u);
                inputs[id].data[STACK_BASE + sp] = nb;
                sp += 1u;
            }
        }
        if cy >= 1u {
            let nb = pt_idx(cx, cy - 1u);
            let c = cell_at(id, nb);
            if c == 0u {
                libs += 1u;
            } else if c == color {
                set_cell(id, nb, 3u);
                inputs[id].data[STACK_BASE + sp] = nb;
                sp += 1u;
            }
        }
        if cy <= 17u {
            let nb = pt_idx(cx, cy + 1u);
            let c = cell_at(id, nb);
            if c == 0u {
                libs += 1u;
            } else if c == color {
                set_cell(id, nb, 3u);
                inputs[id].data[STACK_BASE + sp] = nb;
                sp += 1u;
            }
        }
    }
    return libs;
}

// 将 start 处的 3 标记恢复为 color
fn flood_restore(id: u32, start: u32, color: u32) {
    var sp = 0u;
    inputs[id].data[STACK_BASE + sp] = start;
    sp += 1u;
    set_cell(id, start, color);
    var guard = 0u;
    while sp > 0u && guard < 4000u {
        guard += 1u;
        sp -= 1u;
        let cur = inputs[id].data[STACK_BASE + sp];
        let cx = cur % 19u;
        let cy = cur / 19u;
        if cx >= 1u {
            let nb = pt_idx(cx - 1u, cy);
            if cell_at(id, nb) == 3u {
                set_cell(id, nb, color);
                inputs[id].data[STACK_BASE + sp] = nb;
                sp += 1u;
            }
        }
        if cx <= 17u {
            let nb = pt_idx(cx + 1u, cy);
            if cell_at(id, nb) == 3u {
                set_cell(id, nb, color);
                inputs[id].data[STACK_BASE + sp] = nb;
                sp += 1u;
            }
        }
        if cy >= 1u {
            let nb = pt_idx(cx, cy - 1u);
            if cell_at(id, nb) == 3u {
                set_cell(id, nb, color);
                inputs[id].data[STACK_BASE + sp] = nb;
                sp += 1u;
            }
        }
        if cy <= 17u {
            let nb = pt_idx(cx, cy + 1u);
            if cell_at(id, nb) == 3u {
                set_cell(id, nb, color);
                inputs[id].data[STACK_BASE + sp] = nb;
                sp += 1u;
            }
        }
    }
}

// 提掉 start 所在 color 链（先标记 3 再转为 0），返回提子数
fn remove_chain(id: u32, start: u32, color: u32) -> u32 {
    _ = flood_mark(id, start, color);
    var sp = 0u;
    inputs[id].data[STACK_BASE + sp] = start;
    sp += 1u;
    set_cell(id, start, 0u);
    var count = 1u;
    var guard = 0u;
    while sp > 0u && guard < 4000u {
        guard += 1u;
        sp -= 1u;
        let cur = inputs[id].data[STACK_BASE + sp];
        let cx = cur % 19u;
        let cy = cur / 19u;
        if cx >= 1u {
            let nb = pt_idx(cx - 1u, cy);
            if cell_at(id, nb) == 3u {
                set_cell(id, nb, 0u);
                count += 1u;
                inputs[id].data[STACK_BASE + sp] = nb;
                sp += 1u;
            }
        }
        if cx <= 17u {
            let nb = pt_idx(cx + 1u, cy);
            if cell_at(id, nb) == 3u {
                set_cell(id, nb, 0u);
                count += 1u;
                inputs[id].data[STACK_BASE + sp] = nb;
                sp += 1u;
            }
        }
        if cy >= 1u {
            let nb = pt_idx(cx, cy - 1u);
            if cell_at(id, nb) == 3u {
                set_cell(id, nb, 0u);
                count += 1u;
                inputs[id].data[STACK_BASE + sp] = nb;
                sp += 1u;
            }
        }
        if cy <= 17u {
            let nb = pt_idx(cx, cy + 1u);
            if cell_at(id, nb) == 3u {
                set_cell(id, nb, 0u);
                count += 1u;
                inputs[id].data[STACK_BASE + sp] = nb;
                sp += 1u;
            }
        }
    }
    return count;
}

// 链的气数（标记后恢复）
fn chain_liberties(id: u32, start: u32, color: u32) -> u32 {
    let libs = flood_mark(id, start, color);
    flood_restore(id, start, color);
    return libs;
}

// ---- 走子策略（局部启发式）----

// 单子吃子代理：对方子 s 仅 1 个空邻且该空邻就是 p
fn nb_single_liberty_here(id: u32, s: u32, p: u32) -> bool {
    let sx = s % 19u;
    let sy = s / 19u;
    var empties = 0u;
    var the_lib = NO_KO;
    if sx >= 1u {
        let q = pt_idx(sx - 1u, sy);
        if cell_at(id, q) == 0u {
            empties += 1u;
            the_lib = q;
        }
    }
    if sx <= 17u {
        let q = pt_idx(sx + 1u, sy);
        if cell_at(id, q) == 0u {
            empties += 1u;
            the_lib = q;
        }
    }
    if sy >= 1u {
        let q = pt_idx(sx, sy - 1u);
        if cell_at(id, q) == 0u {
            empties += 1u;
            the_lib = q;
        }
    }
    if sy <= 17u {
        let q = pt_idx(sx, sy + 1u);
        if cell_at(id, q) == 0u {
            empties += 1u;
            the_lib = q;
        }
    }
    return empties == 1u && the_lib == p;
}

// 候选点打分：极高负分表示非法（自杀/填眼）
fn evaluate_move(id: u32, p: u32, color: u32) -> i32 {
    let px = p % 19u;
    let py = p / 19u;
    var opp_n = 0u;
    var own_n = 0u;
    var can_cap = false;
    if px >= 1u {
        let nb = pt_idx(px - 1u, py);
        let c = cell_at(id, nb);
        if c == color {
            own_n += 1u;
        } else if c != 0u {
            opp_n += 1u;
            if nb_single_liberty_here(id, nb, p) {
                can_cap = true;
            }
        }
    }
    if px <= 17u {
        let nb = pt_idx(px + 1u, py);
        let c = cell_at(id, nb);
        if c == color {
            own_n += 1u;
        } else if c != 0u {
            opp_n += 1u;
            if nb_single_liberty_here(id, nb, p) {
                can_cap = true;
            }
        }
    }
    if py >= 1u {
        let nb = pt_idx(px, py - 1u);
        let c = cell_at(id, nb);
        if c == color {
            own_n += 1u;
        } else if c != 0u {
            opp_n += 1u;
            if nb_single_liberty_here(id, nb, p) {
                can_cap = true;
            }
        }
    }
    if py <= 17u {
        let nb = pt_idx(px, py + 1u);
        let c = cell_at(id, nb);
        if c == color {
            own_n += 1u;
        } else if c != 0u {
            opp_n += 1u;
            if nb_single_liberty_here(id, nb, p) {
                can_cap = true;
            }
        }
    }
    // 自杀：四邻全为对方且不吃子
    if opp_n == 4u && own_n == 0u && !can_cap {
        return -100000;
    }
    // 填己方眼
    if own_n == 4u {
        return -50000;
    }
    var score = 10i + i32(rng_next() % 11u);
    if can_cap {
        score += 100i;
    }
    if opp_n >= 1u {
        score += 15i;
    }
    if own_n >= 1u {
        score += 8i;
    }
    return score;
}

// 候选加入（去重；存于栈区，返回新数量）
fn cand_try_add(id: u32, p: u32, n: u32) -> u32 {
    if cell_at(id, p) != 0u {
        return n;
    }
    var k = 0u;
    while k < n {
        if inputs[id].data[STACK_BASE + k] == p {
            return n;
        }
        k += 1u;
    }
    if n >= CAND_MAX {
        return n;
    }
    inputs[id].data[STACK_BASE + n] = p;
    return n + 1u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if id >= arrayLength(&inputs) {
        return;
    }
    rng_state = params.seed ^ (id * 2654435761u) ^ (id * id * 2246822519u);

    let meta_info = inputs[id].data[24u];
    var turn = meta_info & 3u;
    var ko = (meta_info >> 2u) & 511u;
    if turn != 1u && turn != 2u {
        turn = 1u;
    }
    var passes = 0u;
    var plies = 0u;

    loop {
        if plies >= MAX_PLIES {
            break;
        }
        // 收集候选：棋子邻点优先，不足补随机空点
        var n_cand = 0u;
        var i = 0u;
        while i < TOTAL {
            if cell_at(id, i) != 0u {
                let x = i % 19u;
                let y = i / 19u;
                if x >= 1u {
                    n_cand = cand_try_add(id, pt_idx(x - 1u, y), n_cand);
                }
                if x <= 17u {
                    n_cand = cand_try_add(id, pt_idx(x + 1u, y), n_cand);
                }
                if y >= 1u {
                    n_cand = cand_try_add(id, pt_idx(x, y - 1u), n_cand);
                }
                if y <= 17u {
                    n_cand = cand_try_add(id, pt_idx(x, y + 1u), n_cand);
                }
            }
            i += 1u;
        }
        var guard = 0u;
        while n_cand < 12u && guard < 300u {
            guard += 1u;
            n_cand = cand_try_add(id, rand_range(TOTAL), n_cand);
        }
        if n_cand == 0u {
            // 无候选 → 虚着
            passes += 1u;
            if passes >= 2u {
                break;
            }
            ko = NO_KO;
            turn = 3u - turn;
            plies += 1u;
            continue;
        }
        // 打分选最优
        var best_idx = 0u;
        var best_score = -1000000i;
        var found = false;
        var k = 0u;
        while k < n_cand {
            let p = inputs[id].data[STACK_BASE + k];
            if p != ko {
                let sc = evaluate_move(id, p, turn);
                if sc > best_score {
                    best_score = sc;
                    best_idx = k;
                    found = true;
                }
            }
            k += 1u;
        }
        if !found {
            passes += 1u;
            if passes >= 2u {
                break;
            }
            ko = NO_KO;
            turn = 3u - turn;
            plies += 1u;
            continue;
        }
        let p = inputs[id].data[STACK_BASE + best_idx];
        passes = 0u;
        // 落子
        set_cell(id, p, turn);
        // 提掉无气对方链
        var captured = 0u;
        var single_pos = NO_KO;
        let px = p % 19u;
        let py = p / 19u;
        if px >= 1u {
            let nb = pt_idx(px - 1u, py);
            if cell_at(id, nb) == 3u - turn && chain_liberties(id, nb, 3u - turn) == 0u {
                let cnt = remove_chain(id, nb, 3u - turn);
                captured += cnt;
                if cnt == 1u {
                    single_pos = nb;
                }
            }
        }
        if px <= 17u {
            let nb = pt_idx(px + 1u, py);
            if cell_at(id, nb) == 3u - turn && chain_liberties(id, nb, 3u - turn) == 0u {
                let cnt = remove_chain(id, nb, 3u - turn);
                captured += cnt;
                if cnt == 1u {
                    single_pos = nb;
                }
            }
        }
        if py >= 1u {
            let nb = pt_idx(px, py - 1u);
            if cell_at(id, nb) == 3u - turn && chain_liberties(id, nb, 3u - turn) == 0u {
                let cnt = remove_chain(id, nb, 3u - turn);
                captured += cnt;
                if cnt == 1u {
                    single_pos = nb;
                }
            }
        }
        if py <= 17u {
            let nb = pt_idx(px, py + 1u);
            if cell_at(id, nb) == 3u - turn && chain_liberties(id, nb, 3u - turn) == 0u {
                let cnt = remove_chain(id, nb, 3u - turn);
                captured += cnt;
                if cnt == 1u {
                    single_pos = nb;
                }
            }
        }
        // 自杀检查：己方链无气且未提子 → 撤销落子，视为虚着
        if chain_liberties(id, p, turn) == 0u {
            set_cell(id, p, 0u);
            ko = NO_KO;
            passes += 1u;
            if passes >= 2u {
                break;
            }
            turn = 3u - turn;
            plies += 1u;
            continue;
        }
        ko = select(NO_KO, single_pos, captured == 1u);
        turn = 3u - turn;
        plies += 1u;
    }

    // 终局计分（中国规则近似：子数 + 围空，黑贴 3.75 子）
    var b_stones = 0u;
    var w_stones = 0u;
    var idx2 = 0u;
    while idx2 < TOTAL {
        let c = cell_at(id, idx2);
        if c == 1u {
            b_stones += 1u;
        } else if c == 2u {
            w_stones += 1u;
        }
        idx2 += 1u;
    }
    var b_terr = 0u;
    var w_terr = 0u;
    var start = 0u;
    while start < TOTAL {
        if cell_at(id, start) != 0u {
            start += 1u;
            continue;
        }
        // 洪泛空区域（标记 3，结束后统一恢复）
        var touches_b = false;
        var touches_w = false;
        var region = 0u;
        var sp = 0u;
        inputs[id].data[STACK_BASE + sp] = start;
        sp += 1u;
        set_cell(id, start, 3u);
        var guard2 = 0u;
        while sp > 0u && guard2 < 4000u {
            guard2 += 1u;
            sp -= 1u;
            let cur = inputs[id].data[STACK_BASE + sp];
            region += 1u;
            let cx = cur % 19u;
            let cy = cur / 19u;
            if cx >= 1u {
                let nb = pt_idx(cx - 1u, cy);
                let c = cell_at(id, nb);
                if c == 0u {
                    set_cell(id, nb, 3u);
                    inputs[id].data[STACK_BASE + sp] = nb;
                    sp += 1u;
                } else if c == 1u {
                    touches_b = true;
                } else if c == 2u {
                    touches_w = true;
                }
            }
            if cx <= 17u {
                let nb = pt_idx(cx + 1u, cy);
                let c = cell_at(id, nb);
                if c == 0u {
                    set_cell(id, nb, 3u);
                    inputs[id].data[STACK_BASE + sp] = nb;
                    sp += 1u;
                } else if c == 1u {
                    touches_b = true;
                } else if c == 2u {
                    touches_w = true;
                }
            }
            if cy >= 1u {
                let nb = pt_idx(cx, cy - 1u);
                let c = cell_at(id, nb);
                if c == 0u {
                    set_cell(id, nb, 3u);
                    inputs[id].data[STACK_BASE + sp] = nb;
                    sp += 1u;
                } else if c == 1u {
                    touches_b = true;
                } else if c == 2u {
                    touches_w = true;
                }
            }
            if cy <= 17u {
                let nb = pt_idx(cx, cy + 1u);
                let c = cell_at(id, nb);
                if c == 0u {
                    set_cell(id, nb, 3u);
                    inputs[id].data[STACK_BASE + sp] = nb;
                    sp += 1u;
                } else if c == 1u {
                    touches_b = true;
                } else if c == 2u {
                    touches_w = true;
                }
            }
        }
        if touches_b && !touches_w {
            b_terr += region;
        } else if touches_w && !touches_b {
            w_terr += region;
        }
        start += 1u;
    }
    // 恢复所有 3 标记为空
    var j = 0u;
    while j < TOTAL {
        if cell_at(id, j) == 3u {
            set_cell(id, j, 0u);
        }
        j += 1u;
    }
    // 中国规则：黑总 - 3.75 vs 白总（×4 取整比较）
    let b_total = b_stones + b_terr;
    let w_total = w_stones + w_terr;
    let d = i32(b_total * 4u) - 15 - i32(w_total * 4u);
    var winner = 2u;
    if d > 0 {
        winner = 0u; // 黑胜
    } else if d < 0 {
        winner = 1u; // 白胜
    }
    outputs[id].winner = winner;
}
