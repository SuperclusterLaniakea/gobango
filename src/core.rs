//! 核心规则模块：棋盘、棋链、提子、禁着、简单劫（Zobrist）、对局状态与中国规则计分。
//!
//! 数据表示采用 1D 数组加边界哨兵：(19+2)×(19+2)，外围一圈始终为 `Empty` 且不可落子，
//! 从而简化邻居遍历与区域洪水填充（触摸棋盘边缘的空域视为中立）。

use std::collections::HashSet;
use std::sync::OnceLock;

/// 棋盘路数（19 路）。
pub const BOARD_SIZE: usize = 19;
/// 中国规则黑贴 3.75 子（即 7.5 目）。
pub const KOMI: f64 = 3.75;
/// 对局内保留的局面快照数量上限（神经网络只用最近 4 个，多留几个便于调试与扩展）。
pub const SNAP_KEEP: usize = 8;
/// 1D 数组步长（含左右哨兵列）。
const STRIDE: usize = BOARD_SIZE + 2;
/// 1D 数组总长（含外围哨兵）。
const GRID_LEN: usize = STRIDE * STRIDE;

/// 棋子颜色。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Color {
    Empty,
    Black,
    White,
}

impl Color {
    /// 对手颜色。
    pub fn opponent(self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
            Color::Empty => Color::Empty,
        }
    }

    /// 中文名称。
    pub fn label(self) -> &'static str {
        match self {
            Color::Black => "黑棋",
            Color::White => "白棋",
            Color::Empty => "空",
        }
    }

    /// GTP/SGF 单字母。
    pub fn sgf(self) -> char {
        match self {
            Color::Black => 'B',
            Color::White => 'W',
            Color::Empty => '?',
        }
    }
}

/// 棋盘坐标：`x` 为列（0..19，左→右），`y` 为行（0..19，上→下）。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Point {
    pub x: usize,
    pub y: usize,
}

impl Point {
    pub fn new(x: usize, y: usize) -> Self {
        Point { x, y }
    }

    /// 是否落在 19×19 盘面内。
    pub fn in_bounds(self) -> bool {
        self.x < BOARD_SIZE && self.y < BOARD_SIZE
    }

    /// SGF 坐标，如 (16, 3) → "Q16"。字母跳过 'I'。
    pub fn sgf(self) -> String {
        let letter = if self.x < 8 { b'A' + self.x as u8 } else { b'A' + self.x as u8 + 1 };
        format!("{}{}", letter as char, BOARD_SIZE - self.y)
    }

    /// 解析 SGF 坐标："Q16"、"a1"、"PASS" 之外的点。失败返回 None。
    pub fn from_sgf(s: &str) -> Option<Point> {
        let s = s.trim();
        if s.is_empty() || s.len() < 2 {
            return None;
        }
        let bytes = s.as_bytes();
        let mut letter = bytes[0].to_ascii_uppercase();
        if !(b'A'..=b'T').contains(&letter) || letter == b'I' {
            return None;
        }
        if letter > b'I' {
            letter -= 1; // 去掉 I，恢复 0..18 列索引
        }
        let x = (letter - b'A') as usize;
        let num: usize = s[1..].trim().parse().ok()?;
        if num < 1 || num > BOARD_SIZE {
            return None;
        }
        let y = BOARD_SIZE - num;
        if x >= BOARD_SIZE || y >= BOARD_SIZE {
            return None;
        }
        Some(Point { x, y })
    }
}

/// 着法：落子 / 虚着 / 认输。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Move {
    Pass,
    Resign,
    Play(Point),
}

impl Move {
    pub fn is_pass(&self) -> bool {
        matches!(self, Move::Pass)
    }
    pub fn is_resign(&self) -> bool {
        matches!(self, Move::Resign)
    }
    pub fn is_play(&self) -> bool {
        matches!(self, Move::Play(_))
    }

    /// SGF 表示，如 "B Q16" / "W pass" / "B resign"。
    pub fn sgf(&self, color: Color) -> String {
        match self {
            Move::Pass => format!("{} pass", color.sgf()),
            Move::Resign => format!("{} resign", color.sgf()),
            Move::Play(p) => format!("{} {}", color.sgf(), p.sgf()),
        }
    }
}

/// Zobrist 哈希表：每个 (颜色, 点) 一个 64 位随机数 + 回合随机数。
/// 使用固定种子的 splitmix64 生成，保证跨运行确定性（利于测试复现）。
struct Zobrist {
    stones: [[u64; BOARD_SIZE * BOARD_SIZE]; 2],
    turn: u64,
}

static ZOBRIST: OnceLock<Zobrist> = OnceLock::new();

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn zobrist() -> &'static Zobrist {
    ZOBRIST.get_or_init(|| {
        let mut state: u64 = 0x9E3779B97F4A7C15 ^ 0x1234_5678_9ABC_DEF0;
        let mut stones = [[0u64; BOARD_SIZE * BOARD_SIZE]; 2];
        for color in 0..2 {
            for cell in 0..BOARD_SIZE * BOARD_SIZE {
                stones[color][cell] = splitmix64(&mut state);
            }
        }
        let turn = splitmix64(&mut state);
        Zobrist { stones, turn }
    })
}

/// 棋盘：1D 数组 + 哨兵环。
#[derive(Clone)]
pub struct Board {
    grid: [Color; GRID_LEN],
    /// 仅棋子（不含回合）的 Zobrist 哈希。
    stone_hash: u64,
    /// 当前行棋方。
    turn: Color,
    /// 简单劫点：上一手单子提子后，禁止下一手立即回提的点。
    ko_point: Option<Point>,
    black_captures: u32,
    white_captures: u32,
    /// 最近一手提子数（供快速走子策略评分）。
    pub last_captured: u32,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    /// 新建空棋盘，黑先。
    pub fn new() -> Self {
        Board {
            grid: [Color::Empty; GRID_LEN],
            stone_hash: 0,
            turn: Color::Black,
            ko_point: None,
            black_captures: 0,
            white_captures: 0,
            last_captured: 0,
        }
    }

    /// (x, y) → 1D 索引（含哨兵环偏移）。
    #[inline]
    fn idx(x: usize, y: usize) -> usize {
        (y + 1) * STRIDE + (x + 1)
    }

    /// 当前行棋方。
    pub fn turn(&self) -> Color {
        self.turn
    }

    /// 该点棋子颜色（不含哨兵环语义，环上永远返回 Empty）。
    pub fn at(&self, x: usize, y: usize) -> Color {
        self.grid[Self::idx(x, y)]
    }

    pub fn at_point(&self, p: Point) -> Color {
        self.at(p.x, p.y)
    }

    /// 该点是否为空（且可落子，哨兵环返回 false）。
    pub fn is_empty(&self, x: usize, y: usize) -> bool {
        x < BOARD_SIZE && y < BOARD_SIZE && self.grid[Self::idx(x, y)] == Color::Empty
    }

    /// 提子数。
    pub fn black_captures(&self) -> u32 {
        self.black_captures
    }
    pub fn white_captures(&self) -> u32 {
        self.white_captures
    }

    /// 劫点（若有）。
    pub fn ko_point(&self) -> Option<Point> {
        self.ko_point
    }

    /// 完整局面哈希（棋子 + 回合），供同形判断与调试。
    pub fn hash(&self) -> u64 {
        let turn_bit = if self.turn == Color::Black { 0 } else { zobrist().turn };
        self.stone_hash ^ turn_bit
    }

    /// 仅棋子哈希。
    pub fn stone_hash(&self) -> u64 {
        self.stone_hash
    }

    /// 直接放置棋子（仅供测试与构造局面用，不校验规则、不更新劫/提子/回合）。
    pub fn set(&mut self, x: usize, y: usize, c: Color) {
        if !(x < BOARD_SIZE && y < BOARD_SIZE) {
            return;
        }
        let i = Self::idx(x, y);
        let old = self.grid[i];
        self.grid[i] = c;
        if c != Color::Empty {
            let color_idx = if c == Color::Black { 0 } else { 1 };
            self.stone_hash ^= zobrist().stones[color_idx][y * BOARD_SIZE + x];
        }
        if old != Color::Empty {
            let color_idx = if old == Color::Black { 0 } else { 1 };
            self.stone_hash ^= zobrist().stones[color_idx][y * BOARD_SIZE + x];
        }
    }

    /// 清除劫点（虚着后必须清除）。
    pub fn clear_ko(&mut self) {
        self.ko_point = None;
    }

    /// 统计某色棋子数量。
    pub fn count_stones(&self, c: Color) -> usize {
        let mut n = 0;
        for y in 0..BOARD_SIZE {
            for x in 0..BOARD_SIZE {
                if self.at(x, y) == c {
                    n += 1;
                }
            }
        }
        n
    }

    /// 以 (x, y) 处棋子的棋链为单位，洪水填充统计其气数。
    /// `visited` 用于记录已访问的棋子点。
    fn count_liberties(&self, x: usize, y: usize, visited: &mut HashSet<(usize, usize)>) -> usize {
        let color = self.at(x, y);
        if color == Color::Empty {
            return 0;
        }
        let mut stack = vec![(x, y)];
        let mut liberties = HashSet::new();
        while let Some((cx, cy)) = stack.pop() {
            if !visited.insert((cx, cy)) {
                continue;
            }
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx < 0 || ny < 0 || nx >= BOARD_SIZE as i32 || ny >= BOARD_SIZE as i32 {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                match self.at(nx, ny) {
                    Color::Empty => {
                        liberties.insert((nx, ny));
                    }
                    c if c == color => stack.push((nx, ny)),
                    _ => {}
                }
            }
        }
        liberties.len()
    }

    /// 提掉 (x, y) 处整条棋链，返回提子数；同步更新 Zobrist 哈希。
    fn remove_group(&mut self, x: usize, y: usize) -> usize {
        let color = self.at(x, y);
        if color == Color::Empty {
            return 0;
        }
        let mut stack = vec![(x, y)];
        let mut visited = HashSet::new();
        let mut removed = 0;
        while let Some((cx, cy)) = stack.pop() {
            if !visited.insert((cx, cy)) {
                continue;
            }
            let color_idx = if color == Color::Black { 0 } else { 1 };
            self.stone_hash ^= zobrist().stones[color_idx][cy * BOARD_SIZE + cx];
            self.grid[Self::idx(cx, cy)] = Color::Empty;
            removed += 1;
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx < 0 || ny < 0 || nx >= BOARD_SIZE as i32 || ny >= BOARD_SIZE as i32 {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                if self.at(nx, ny) == color {
                    stack.push((nx, ny));
                }
            }
        }
        removed
    }

    /// 落子（黑/白轮流，黑先）。完整规则：空点、提子、禁自杀、简单劫。
    /// 成功返回 true；失败不改变局面。
    pub fn play(&mut self, x: usize, y: usize) -> bool {
        if !(x < BOARD_SIZE && y < BOARD_SIZE) {
            return false;
        }
        if self.at(x, y) != Color::Empty {
            return false;
        }
        if self.ko_point == Some(Point::new(x, y)) {
            return false;
        }
        let color = self.turn;
        let opponent = color.opponent();

        // 快照，便于回滚
        let saved_grid = self.grid;
        let saved_hash = self.stone_hash;
        let saved_captures = (self.black_captures, self.white_captures);

        self.set(x, y, color);
        self.last_captured = 0;

        // 提掉无气的对方棋链
        let mut captured_total = 0u32;
        let mut single_capture_pos = None;
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= BOARD_SIZE as i32 || ny >= BOARD_SIZE as i32 {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            if self.at(nx, ny) == opponent {
                let mut visited = HashSet::new();
                if self.count_liberties(nx, ny, &mut visited) == 0 {
                    let k = self.remove_group(nx, ny);
                    captured_total += k as u32;
                    if k == 1 {
                        single_capture_pos = Some(Point::new(nx, ny));
                    }
                }
            }
        }
        self.last_captured = captured_total;

        // 禁自杀：己方棋链无气（且未提子）
        let mut visited = HashSet::new();
        if self.count_liberties(x, y, &mut visited) == 0 {
            self.grid = saved_grid;
            self.stone_hash = saved_hash;
            self.black_captures = saved_captures.0;
            self.white_captures = saved_captures.1;
            self.last_captured = 0;
            return false;
        }

        // 简单劫：仅提一子时设置劫点，禁止立即回提
        self.ko_point = if captured_total == 1 { single_capture_pos } else { None };

        if color == Color::Black {
            self.black_captures += captured_total;
        } else {
            self.white_captures += captured_total;
        }
        self.turn = opponent;
        true
    }

    /// 判断 (x, y) 是否合法（在副本上模拟落子）。
    pub fn is_legal(&self, x: usize, y: usize) -> bool {
        if !(x < BOARD_SIZE && y < BOARD_SIZE) {
            return false;
        }
        if self.at(x, y) != Color::Empty {
            return false;
        }
        if self.ko_point == Some(Point::new(x, y)) {
            return false;
        }
        let mut copy = self.clone();
        copy.play(x, y)
    }

    /// 枚举全部合法落子点。
    pub fn legal_moves(&self) -> Vec<Point> {
        let mut out = Vec::new();
        for y in 0..BOARD_SIZE {
            for x in 0..BOARD_SIZE {
                if self.is_legal(x, y) {
                    out.push(Point::new(x, y));
                }
            }
        }
        out
    }

    /// 该棋链的空邻点（气）集合。
    pub fn group_liberties(&self, x: usize, y: usize) -> HashSet<(usize, usize)> {
        let color = self.at(x, y);
        let mut stack = vec![(x, y)];
        let mut visited = HashSet::new();
        let mut liberties = HashSet::new();
        while let Some((cx, cy)) = stack.pop() {
            if !visited.insert((cx, cy)) {
                continue;
            }
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx < 0 || ny < 0 || nx >= BOARD_SIZE as i32 || ny >= BOARD_SIZE as i32 {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                match self.at(nx, ny) {
                    Color::Empty => {
                        liberties.insert((nx, ny));
                    }
                    c if c == color => stack.push((nx, ny)),
                    _ => {}
                }
            }
        }
        liberties
    }

    /// 该棋链的气数（公开接口，供 AI 启发式使用）。
    pub fn liberties_count(&self, x: usize, y: usize) -> usize {
        let mut visited = HashSet::new();
        self.count_liberties(x, y, &mut visited)
    }

    /// 该棋链的全部棋子点。
    pub fn group_points(&self, x: usize, y: usize) -> HashSet<(usize, usize)> {
        let color = self.at(x, y);
        let mut stack = vec![(x, y)];
        let mut visited = HashSet::new();
        while let Some((cx, cy)) = stack.pop() {
            if !visited.insert((cx, cy)) {
                continue;
            }
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx < 0 || ny < 0 || nx >= BOARD_SIZE as i32 || ny >= BOARD_SIZE as i32 {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                if self.at(nx, ny) == color {
                    stack.push((nx, ny));
                }
            }
        }
        visited
    }

    /// 空点 p 是否为颜色 c 的真眼（按说明书 5.7.1 的简化判定）。
    /// 四个正交邻居必须全为 c；角/边/中央的对角线控制数分别 ≥1/≥2/≥3。
    fn is_true_eye(&self, p: (usize, usize), c: Color) -> bool {
        let (x, y) = p;
        if !(x < BOARD_SIZE && y < BOARD_SIZE) {
            return false;
        }
        if self.at(x, y) != Color::Empty {
            return false;
        }
        // 正交邻居必须全是 c
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= BOARD_SIZE as i32 || ny >= BOARD_SIZE as i32 {
                continue; // 邻边视为已控制（边界）
            }
            if self.at(nx as usize, ny as usize) != c {
                return false;
            }
        }
        // 对角线控制数（出界视为已控制）
        let mut owned = 0;
        for (dx, dy) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= BOARD_SIZE as i32 || ny >= BOARD_SIZE as i32 {
                owned += 1;
            } else if self.at(nx as usize, ny as usize) == c {
                owned += 1;
            }
        }
        let on_edge = x == 0 || y == 0 || x == BOARD_SIZE - 1 || y == BOARD_SIZE - 1;
        let corner = (x == 0 || x == BOARD_SIZE - 1) && (y == 0 || y == BOARD_SIZE - 1);
        if corner {
            owned >= 1
        } else if on_edge {
            owned >= 2
        } else {
            owned >= 3
        }
    }

    /// 自动标记死子：某棋链若无真眼（简化判定）且气数 ≤ 2，视为死子。
    /// 该启发式对复杂死活可能不准确，可配合手动标记修正（见 `Game::mark_dead`）。
    pub fn auto_dead_points(&self) -> HashSet<(usize, usize)> {
        let mut dead = HashSet::new();
        let mut visited = HashSet::new();
        for y in 0..BOARD_SIZE {
            for x in 0..BOARD_SIZE {
                let c = self.at(x, y);
                if c == Color::Empty || visited.contains(&(x, y)) {
                    continue;
                }
                let group = self.group_points(x, y);
                for &p in &group {
                    visited.insert(p);
                }
                let liberties = self.group_liberties(x, y);
                if liberties.len() > 2 {
                    continue; // 空间足够，不自动判死
                }
                let eyes = liberties
                    .iter()
                    .filter(|&&p| self.is_true_eye(p, c))
                    .count();
                if eyes == 0 {
                    dead.extend(group);
                }
            }
        }
        dead
    }
}

/// 对局状态。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameStatus {
    /// 对局进行中。
    Ongoing,
    /// 双方连续虚着两次，进入计分阶段（可标记死子）。
    Scoring,
    /// 已结束（计分确认或认输）。
    Finished,
}

/// 中国规则数子法计分结果。
#[derive(Clone, Copy, Debug)]
pub struct Score {
    pub black_stones: usize,
    pub white_stones: usize,
    pub black_territory: usize,
    pub white_territory: usize,
    pub neutral: usize,
    pub dead_removed: usize,
    /// 黑方总子数（盘面黑子 + 黑空）。
    pub black_total: usize,
    /// 白方总子数（盘面白子 + 白空）。
    pub white_total: usize,
    /// 黑方最终得分（扣贴目 3.75 子）。
    pub black_final: f64,
    pub white_final: f64,
    pub winner: Option<Color>,
    /// 胜负差（子）。
    pub margin: f64,
}

/// 按中国规则数子法计分：先移除死子，再对空域洪水填充。
/// 只邻接黑棋的空域为黑空，只邻接白棋的为空白，触摸棋盘边缘或邻接双方为中立。
pub fn compute_score(board: &Board, dead: &HashSet<(usize, usize)>) -> Score {
    // 复制局面并移除死子
    let mut b = board.clone();
    let mut dead_removed = 0;
    for &(x, y) in dead {
        if b.at(x, y) != Color::Empty {
            b.set(x, y, Color::Empty);
            dead_removed += 1;
        }
    }

    let mut black_stones = 0usize;
    let mut white_stones = 0usize;
    let mut black_territory = 0usize;
    let mut white_territory = 0usize;
    let mut neutral = 0usize;

    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            match b.at(x, y) {
                Color::Black => black_stones += 1,
                Color::White => white_stones += 1,
                Color::Empty => {}
            }
        }
    }

    // 空域洪水填充
    let mut visited = vec![vec![false; BOARD_SIZE]; BOARD_SIZE];
    for y in 0..BOARD_SIZE {
        for x in 0..BOARD_SIZE {
            if b.at(x, y) != Color::Empty || visited[y][x] {
                continue;
            }
            let mut region = Vec::new();
            let mut stack = vec![(x, y)];
            let mut touches_black = false;
            let mut touches_white = false;
            let mut touches_edge = false;
            while let Some((cx, cy)) = stack.pop() {
                if visited[cy][cx] {
                    continue;
                }
                visited[cy][cx] = true;
                region.push((cx, cy));
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= BOARD_SIZE as i32 || ny >= BOARD_SIZE as i32 {
                        touches_edge = true;
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    match b.at(nx, ny) {
                        Color::Black => touches_black = true,
                        Color::White => touches_white = true,
                        Color::Empty => {
                            if !visited[ny][nx] {
                                stack.push((nx, ny));
                            }
                        }
                    }
                }
            }
            if touches_edge || (touches_black && touches_white) {
                neutral += region.len();
            } else if touches_black {
                black_territory += region.len();
            } else if touches_white {
                white_territory += region.len();
            } else {
                neutral += region.len();
            }
        }
    }

    let black_total = black_stones + black_territory;
    let white_total = white_stones + white_territory;
    let black_final = black_total as f64 - KOMI;
    let white_final = white_total as f64;
    let (winner, margin) = if (black_final - white_final).abs() < 1e-9 {
        (None, 0.0)
    } else if black_final > white_final {
        (Some(Color::Black), black_final - white_final)
    } else {
        (Some(Color::White), white_final - black_final)
    };

    Score {
        black_stones,
        white_stones,
        black_territory,
        white_territory,
        neutral,
        dead_removed,
        black_total,
        white_total,
        black_final,
        white_final,
        winner,
        margin,
    }
}

/// 一局围棋：管理行棋方、虚着计数、悔棋、死子标记与终局。
#[derive(Clone)]
pub struct Game {
    board: Board,
    history: Vec<(Move, Color)>,
    /// 最近局面快照（末尾 = 当前局面），供神经网络特征构建使用。
    /// 只保留最近若干手，长度受 `SNAP_KEEP` 限制。
    snapshots: std::collections::VecDeque<Board>,
    /// 连续虚着次数。
    passes: u32,
    status: GameStatus,
    winner: Option<Color>,
    dead: HashSet<(usize, usize)>,
    komi: f64,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    pub fn new() -> Self {
        let mut snapshots = std::collections::VecDeque::new();
        snapshots.push_back(Board::new());
        Game {
            board: Board::new(),
            history: Vec::new(),
            snapshots,
            passes: 0,
            status: GameStatus::Ongoing,
            winner: None,
            dead: HashSet::new(),
            komi: KOMI,
        }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn board_mut(&mut self) -> &mut Board {
        &mut self.board
    }

    pub fn current_player(&self) -> Color {
        self.board.turn()
    }

    pub fn status(&self) -> GameStatus {
        self.status
    }

    pub fn winner(&self) -> Option<Color> {
        self.winner
    }

    pub fn komi(&self) -> f64 {
        self.komi
    }

    pub fn set_komi(&mut self, komi: f64) {
        self.komi = komi;
    }

    /// 手数（含 pass/resign）。
    pub fn move_count(&self) -> usize {
        self.history.len()
    }

    /// 完整着法历史（着法, 行棋方），供棋谱序列化与复盘使用。
    pub fn history(&self) -> &[(Move, Color)] {
        &self.history
    }

    /// 已标记死子集合。
    pub fn dead(&self) -> &HashSet<(usize, usize)> {
        &self.dead
    }

    /// 最近一手着法与行棋方（若有）。
    pub fn last_move(&self) -> Option<(Move, Color)> {
        self.history.last().copied()
    }

    /// 落子历史（颜色, 点），供开局库匹配使用。
    pub fn book_history(&self) -> Vec<(Color, Point)> {
        self.history
            .iter()
            .filter_map(|(m, c)| match m {
                Move::Play(p) => Some((*c, *p)),
                _ => None,
            })
            .collect()
    }

    /// 执行一手棋。非法着法返回 false 且不改变局面。
    pub fn play(&mut self, m: Move) -> bool {
        let color = self.current_player();
        match m {
            Move::Resign => {
                self.status = GameStatus::Finished;
                self.winner = Some(color.opponent());
                self.history.push((Move::Resign, color));
                true
            }
            Move::Pass => {
                self.board.clear_ko();
                self.passes += 1;
                self.history.push((Move::Pass, color));
                if self.passes >= 2 {
                    self.status = GameStatus::Scoring;
                    // 进入计分阶段时自动标记死子（供用户修正）
                    self.dead = self.board.auto_dead_points();
                }
                true
            }
            Move::Play(p) => {
                if !self.board.play(p.x, p.y) {
                    return false;
                }
                self.passes = 0;
                self.history.push((Move::Play(p), color));
                self.push_snapshot();
                if self.status == GameStatus::Scoring {
                    self.status = GameStatus::Ongoing;
                }
                true
            }
        }
    }

    /// 悔棋一步（若此前处于计分/结束状态则恢复进行中）。
    /// 返回被撤销的着法。无棋可悔时返回 None。
    pub fn undo(&mut self) -> Option<(Move, Color)> {
        let popped = self.history.pop()?;
        // 从空盘重建局面（同时重建神经网络所需的局面快照）
        self.rebuild_from_history();
        // 重算连续虚着
        self.passes = self
            .history
            .iter()
            .rev()
            .take_while(|(m, _)| m.is_pass())
            .count() as u32;
        self.status = GameStatus::Ongoing;
        self.winner = None;
        self.dead.clear();
        Some(popped)
    }

    /// 从 `history` 重放重建棋盘与快照队列。
    fn rebuild_from_history(&mut self) {
        let mut board = Board::new();
        self.snapshots.clear();
        self.snapshots.push_back(board.clone());
        for &(m, _) in &self.history {
            match m {
                Move::Play(p) => {
                    board.play(p.x, p.y);
                    self.snapshots.push_back(board.clone());
                }
                Move::Pass => {
                    board.clear_ko();
                }
                Move::Resign => break,
            }
        }
        self.board = board;
        while self.snapshots.len() > SNAP_KEEP {
            self.snapshots.pop_front();
        }
    }

    /// 追加当前局面快照（落子后调用），只保留最近 `SNAP_KEEP` 个。
    fn push_snapshot(&mut self) {
        self.snapshots.push_back(self.board.clone());
        while self.snapshots.len() > SNAP_KEEP {
            self.snapshots.pop_front();
        }
    }

    /// 取最近 `n` 个局面快照，**按时间倒序**返回：[0] = 当前局面，[1] = 上一手之前，依此类推。
    ///
    /// 这正是神经网络特征所需的顺序（与训练时 `board.snaps` 一致）。
    pub fn recent_snapshots(&self, n: usize) -> Vec<Board> {
        self.snapshots.iter().rev().take(n).cloned().collect()
    }

    /// 计分阶段手动标记/取消标记死子。
    pub fn mark_dead(&mut self, x: usize, y: usize) -> bool {
        if self.status != GameStatus::Scoring {
            return false;
        }
        let p = (x, y);
        if self.board.at(x, y) == Color::Empty {
            return false;
        }
        if !self.dead.remove(&p) {
            self.dead.insert(p);
        }
        true
    }

    /// 设置死子标记集合（供“判断胜负 → 确认结束”写入判定所用的死子）。
    pub fn set_dead(&mut self, dead: HashSet<(usize, usize)>) {
        self.dead = dead;
    }

    /// 确认终局：计算最终得分。
    pub fn confirm_score(&mut self) -> Score {
        let score = compute_score(&self.board, &self.dead);
        self.winner = score.winner;
        self.status = GameStatus::Finished;
        score
    }

    /// 计算当前（不修改状态）得分。
    pub fn current_score(&self) -> Score {
        compute_score(&self.board, &self.dead)
    }

    /// 重新开始。
    pub fn reset(&mut self) {
        *self = Game::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一个黑棋包围的左上角局面：7×7 黑环 + 中心黑子 + 环内一枚无眼白子。
    fn ring_game() -> Game {
        let mut g = Game::new();
        {
            let b = g.board_mut();
            for x in 0..7 {
                b.set(x, 0, Color::Black);
                b.set(x, 6, Color::Black);
            }
            for y in 1..6 {
                b.set(0, y, Color::Black);
                b.set(6, y, Color::Black);
            }
            b.set(3, 3, Color::Black);
            b.set(1, 1, Color::White);
        }
        g
    }

    // ---------- T001 单子提子 ----------
    #[test]
    fn t001_single_capture() {
        let mut b = Board::new();
        // 黑 (0,0)、(0,2)，白 (0,1)，随后黑 (1,1) 提掉白 (0,1)
        assert!(b.play(0, 0));
        assert!(b.play(0, 1)); // 白
        assert!(b.play(0, 2));
        assert!(b.play(5, 5)); // 白（别处）
        assert!(b.play(1, 1)); // 黑提子
        assert_eq!(b.at(0, 1), Color::Empty, "无气白子应被提掉");
        assert_eq!(b.black_captures(), 1);
        assert_eq!(b.last_captured, 1);
    }

    // ---------- T002 自杀禁手 ----------
    #[test]
    fn t002_suicide_forbidden() {
        let mut b = Board::new();
        assert!(b.play(1, 0)); // 黑
        assert!(b.play(5, 5)); // 白
        assert!(b.play(0, 1)); // 黑，围住 (0,0)
        // 白在 (0,0) 落子会无气且不提子 → 非法
        assert!(!b.play(0, 0), "自杀着法必须非法");
        assert!(!b.is_legal(0, 0));
        assert_eq!(b.at(0, 0), Color::Empty);
        assert_eq!(b.turn(), Color::White, "非法着法不应改变行棋方");
    }

    // ---------- T003 简单劫 ----------
    #[test]
    fn t003_simple_ko() {
        let mut b = Board::new();
        // 白 (1,1) 被黑 (0,1)(1,0)(1,2) 三面包围，仅剩 (2,1) 一气
        assert!(b.play(0, 1)); // 黑
        assert!(b.play(1, 1)); // 白
        assert!(b.play(1, 0)); // 黑
        assert!(b.play(9, 9)); // 白（别处）
        assert!(b.play(1, 2)); // 黑
        assert!(b.play(9, 10)); // 白（别处）
        // 黑 (2,1) 提掉白 (1,1) → 劫点 (1,1)
        assert!(b.play(2, 1));
        assert_eq!(b.ko_point(), Some(Point::new(1, 1)), "单子提子应产生劫点");
        assert_eq!(b.at(1, 1), Color::Empty);
        // 白立即回提 (1,1) → 非法（劫）
        assert!(!b.is_legal(1, 1));
        assert!(!b.play(1, 1));
        assert_eq!(b.turn(), Color::White, "被劫禁的着法不应改变行棋方");
        // 白在别处落子后劫点解除
        assert!(b.play(9, 11));
        assert_eq!(b.ko_point(), None, "劫点在下一手后解除");
    }

    // ---------- T004 连续两次 pass 进入计分 ----------
    #[test]
    fn t004_double_pass_enters_scoring() {
        let mut g = Game::new();
        assert!(g.play(Move::Pass));
        assert_eq!(g.status(), GameStatus::Ongoing, "单次虚着不结束");
        assert!(g.play(Move::Pass));
        assert_eq!(g.status(), GameStatus::Scoring, "连续两次虚着进入计分");
        // 计分阶段再落子则恢复进行中
        assert!(g.play(Move::Play(Point::new(3, 3))));
        assert_eq!(g.status(), GameStatus::Ongoing);
    }

    // ---------- T005 数子法计分 ----------
    #[test]
    fn t005_chinese_scoring() {
        let mut g = ring_game();
        assert!(g.play(Move::Pass));
        assert!(g.play(Move::Pass));
        assert_eq!(g.status(), GameStatus::Scoring);
        // 自动死子标记应命中环内无眼白子
        assert!(g.dead().contains(&(1, 1)), "无眼且气紧的白子应被自动判死");
        let score = g.current_score();
        assert_eq!(score.black_stones, 25, "黑子：环 24 + 中心 1");
        assert_eq!(score.black_territory, 24, "黑空：内区空 23 + 死子点 1");
        assert_eq!(score.white_stones, 0);
        assert_eq!(score.black_total, 49);
        assert_eq!(score.white_total, 0);
        assert!((score.black_final - 45.25).abs() < 1e-9, "黑 49 子扣 3.75 贴目");
        assert_eq!(score.winner, Some(Color::Black));
        assert!((score.margin - 45.25).abs() < 1e-9);
    }

    /// 有两只独立真眼的白棋大龙：不应被自动判死，且两眼计入白空。
    #[test]
    fn t005b_two_eye_group_alive() {
        let mut g = Game::new();
        {
            let b = g.board_mut();
            let whites = [
                // 眼 A (9,9) 的四围
                (8, 9), (9, 8), (10, 9), (9, 10),
                // 连接竖棒
                (9, 11), (9, 12),
                // 眼 B (9,13) 的 3×3 环
                (8, 12), (8, 13), (8, 14), (9, 14), (10, 12), (10, 13), (10, 14),
            ];
            for &(x, y) in &whites {
                b.set(x, y, Color::White);
            }
        }
        let dead = g.board().auto_dead_points();
        for &(x, y) in &[
            (8, 9), (9, 8), (10, 9), (9, 10), (9, 11), (9, 12),
            (8, 12), (8, 13), (8, 14), (9, 14), (10, 12), (10, 13), (10, 14),
        ] {
            assert!(!dead.contains(&(x, y)), "两眼活棋不应被自动判死: ({x},{y})");
        }
        assert!(g.play(Move::Pass));
        assert!(g.play(Move::Pass));
        let score = g.current_score();
        assert_eq!(score.white_stones, 13);
        assert_eq!(score.white_territory, 2, "两眼各 1 点");
        assert_eq!(score.white_total, 15);
        assert_eq!(score.winner, Some(Color::White));
    }

    // ---------- SGF 坐标往返 ----------
    #[test]
    fn sgf_coordinate_roundtrip() {
        for (x, y, sgf) in [
            (0, 0, "A19"),
            (8, 3, "J16"),
            (15, 3, "Q16"),
            (16, 3, "R16"),
            (18, 18, "T1"),
            (9, 9, "K10"),
        ] {
            let p = Point::new(x, y);
            assert_eq!(p.sgf(), sgf);
            assert_eq!(Point::from_sgf(sgf), Some(p));
            assert_eq!(Point::from_sgf(&sgf.to_lowercase()), Some(p));
        }
        assert_eq!(Point::from_sgf("I5"), None, "SGF 跳过 I 列");
        assert_eq!(Point::from_sgf("X1"), None);
        assert_eq!(Point::from_sgf("A20"), None);
    }

    // ---------- 悔棋 ----------
    #[test]
    fn undo_restores_position() {
        let mut g = Game::new();
        assert!(g.play(Move::Play(Point::new(3, 3))));
        assert!(g.play(Move::Play(Point::new(15, 15))));
        assert!(g.play(Move::Play(Point::new(3, 4))));
        assert_eq!(g.move_count(), 3);
        assert!(g.undo().is_some());
        assert_eq!(g.move_count(), 2);
        assert_eq!(g.board().at(3, 4), Color::Empty);
        assert_eq!(g.current_player(), Color::Black, "悔棋后回到黑方行棋");
        // 悔到空盘
        g.undo();
        g.undo();
        assert_eq!(g.move_count(), 0);
        assert!(g.undo().is_none());
    }

    // ---------- 认输 ----------
    #[test]
    fn resign_finishes_game() {
        let mut g = Game::new();
        assert!(g.play(Move::Resign));
        assert_eq!(g.status(), GameStatus::Finished);
        assert_eq!(g.winner(), Some(Color::White), "黑认输则白胜");
    }
}

