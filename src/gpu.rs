//! GPU 批量走子引擎：用 wgpu 计算着色器并行运行随机走子（playout）。
//!
//! 无 GPU/驱动时 `GpuEngine::try_new()` 返回错误，调用方应回退 CPU 走子。
//! 走子逻辑见 `gpu_playout.wgsl`（每线程一局，批量 dispatch，返回各局胜方）。

use crate::core::{Board, Color};
use bytemuck::{Pod, Zeroable};

/// 每个位置的打包 u32 数（棋盘 23 + 元数据 1 + 栈 300）。
pub const INPUT_WORDS: usize = 324;
/// 单批最大并行走子数。
pub const MAX_BATCH: usize = 1024;
/// workgroup 线程数（与着色器 `@workgroup_size(64)` 一致）。
const WORKGROUP: u32 = 64;

/// GPU 走子输入：打包后的局面（见着色器头部注释）。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PlayoutInput {
    pub data: [u32; INPUT_WORDS],
}

// bytemuck 对超大数组（>64 元素）无内置实现，手工标记 Pod/Zeroable（全 u32 字段，安全）。
unsafe impl Pod for PlayoutInput {}
unsafe impl Zeroable for PlayoutInput {}

/// GPU 走子输出：胜方（0=黑胜，1=白胜，2=平局）。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PlayoutOutput {
    pub winner: u32,
}

/// 将内部棋盘打包为 GPU 输入（2bit/格 + 行棋方/劫点元数据）。
pub fn pack_board(board: &Board) -> PlayoutInput {
    let mut data = [0u32; INPUT_WORDS];
    for y in 0..19 {
        for x in 0..19 {
            let idx = y * 19 + x;
            let c = match board.at(x, y) {
                Color::Black => 1u32,
                Color::White => 2u32,
                _ => 0u32,
            };
            let w = idx / 16;
            let s = (idx % 16) * 2;
            data[w] |= c << s;
        }
    }
    let turn = match board.turn() {
        Color::White => 2u32,
        _ => 1u32,
    };
    let ko = match board.ko_point() {
        Some(p) => (p.y * 19 + p.x) as u32,
        None => 511u32,
    };
    data[24] = turn | (ko << 2);
    PlayoutInput { data }
}

/// GPU 引擎：持有设备、队列、计算管线与缓冲。
pub struct GpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    input_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    seed: u32,
}

impl GpuEngine {
    /// 初始化 GPU 引擎；失败返回原因（无适配器/驱动等）。
    pub fn try_new() -> Result<GpuEngine, String> {
        pollster::block_on(Self::init_async())
    }

    async fn init_async() -> Result<GpuEngine, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "未找到可用的 GPU 适配器".to_string())?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gobango-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("GPU 设备初始化失败：{e}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gobango-playout"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_playout.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("playout-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("playout-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("playout-pipe"),
            layout: Some(&pl),
            module: &shader,
            entry_point: "main",
        });

        let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("playout-in"),
            size: (MAX_BATCH * INPUT_WORDS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("playout-out"),
            size: (MAX_BATCH * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("playout-read"),
            size: (MAX_BATCH * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        use wgpu::util::DeviceExt;
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("playout-params"),
            contents: bytemuck::bytes_of(&[0x5EED_2026u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("playout-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        Ok(GpuEngine {
            device,
            queue,
            pipeline,
            bind_group,
            input_buf,
            output_buf,
            read_buf,
            params_buf,
            seed: 0x5EED_2026,
        })
    }

    /// 批量运行走子：`inputs` 长度需 ≤ MAX_BATCH 且与 `outputs` 一致。
    pub fn run_playouts(
        &mut self,
        inputs: &[PlayoutInput],
        outputs: &mut [PlayoutOutput],
    ) -> Result<(), String> {
        let n = inputs.len();
        if n == 0 || n > MAX_BATCH {
            return Err(format!("批大小越界：{n}"));
        }
        if outputs.len() != n {
            return Err("输出长度与输入不一致".to_string());
        }
        // 上传局面与种子（每次 dispatch 递增种子，保证同局面多次走子有差异）
        self.queue.write_buffer(&self.input_buf, 0, bytemuck::cast_slice(inputs));
        self.seed = self.seed.wrapping_add(0x9E37_79B9);
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&[self.seed, 0u32]));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(((n as u32).div_ceil(WORKGROUP)).max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.output_buf,
            0,
            &self.read_buf,
            0,
            (n * 4) as u64,
        );
        self.queue.submit([encoder.finish()]);

        // 读回结果
        let slice = self.read_buf.slice(0..(n * 4) as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| "GPU 读回通道错误".to_string())?
            .map_err(|e| format!("GPU 缓冲映射失败：{e}"))?;
        {
            let mapped = slice.get_mapped_range();
            let out: &[PlayoutOutput] = bytemuck::cast_slice(&mapped);
            outputs[..n].copy_from_slice(&out[..n]);
        }
        self.read_buf.unmap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 离线校验 WGSL 着色器语法与语义（无需 GPU）。
    #[test]
    fn wgsl_shader_validates() {
        let src = include_str!("gpu_playout.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("WGSL 语法解析失败");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module).expect("WGSL 语义校验失败");
    }

    /// 打包往返一致性。
    #[test]
    fn pack_board_roundtrip() {
        let mut b = Board::new();
        b.set(3, 3, Color::Black);
        b.set(15, 15, Color::White);
        let packed = pack_board(&b);
        // 解包检查若干格
        for (x, y, expect) in [(3usize, 3usize, 1u32), (15, 15, 2), (0, 0, 0), (9, 9, 0)] {
            let idx = y * 19 + x;
            let c = (packed.data[idx / 16] >> ((idx % 16) * 2)) & 3;
            assert_eq!(c, expect, "({x},{y})");
        }
        let meta = packed.data[24];
        assert_eq!(meta & 3, 1, "黑先");
        assert_eq!((meta >> 2) & 511, 511, "无劫点");
    }

    /// GPU 可用时冒烟测试（无 GPU 则跳过，始终通过）。
    #[test]
    fn gpu_smoke_if_available() {
        let Ok(mut engine) = GpuEngine::try_new() else {
            eprintln!("[gpu] 无可用 GPU，跳过冒烟测试");
            return;
        };
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        // 空盘黑先 × 8
        for _ in 0..8 {
            inputs.push(pack_board(&Board::new()));
            outputs.push(PlayoutOutput { winner: 99 });
        }
        engine.run_playouts(&inputs, &mut outputs).unwrap();
        for o in &outputs {
            assert!(o.winner <= 2, "胜方编码非法：{}", o.winner);
        }
    }

    /// GPU 可用时：整条搜索链路（MCTS + GPU 批量走子）返回合法着法（无 GPU 则跳过）。
    #[test]
    fn gpu_search_integration_if_available() {
        let Ok(engine) = GpuEngine::try_new() else {
            eprintln!("[gpu] 无可用 GPU，跳过集成测试");
            return;
        };
        let engine = std::sync::Arc::new(std::sync::Mutex::new(engine));
        let mut game = crate::core::Game::new();
        for (x, y) in [(3usize, 3usize), (15, 15), (3, 15), (15, 3)] {
            game.play(crate::core::Move::Play(crate::core::Point::new(x, y)));
        }
        let book = crate::ai::OpeningBook::load();
        let cfg = crate::ai::SearchConfig {
            max_time: std::time::Duration::from_secs(30),
            max_sims: 512,
            min_sims: 512,
            threads: 2,
            c_uct: 1.5,
            max_rollout_moves: 60,
            use_book: false,
            book_plies: 0,
            heuristic: 3,
            max_candidates: 40,
            use_patterns: false,
            use_rave: false,
            use_enhanced: false,
            use_virtual_loss: false,
            use_virtual_roll: false,
            root_temperature: 1.0,
        };
        let t0 = std::time::Instant::now();
        let r = crate::ai::select_move_cfg(
            game.board(),
            &game.book_history(),
            &cfg,
            &book,
            0x6F0_2026u64,
            Some(engine),
        );
        let gpu_elapsed = t0.elapsed();

        // 同配置 CPU 走子对照（衡量 GPU 对“下棋计算”的加速）
        let t1 = std::time::Instant::now();
        let _r_cpu = crate::ai::select_move_cfg(
            game.board(),
            &game.book_history(),
            &cfg,
            &book,
            0x6F0_2026u64,
            None,
        );
        let cpu_elapsed = t1.elapsed();
        eprintln!(
            "[gpu] 下棋计算耗时：GPU {:?} vs CPU {:?}（512 模拟 × 2 线程）",
            gpu_elapsed, cpu_elapsed
        );
        assert!(
            gpu_elapsed < std::time::Duration::from_secs(30),
            "GPU 搜索耗时异常：{gpu_elapsed:?}"
        );
        match r.mv {
            crate::core::Move::Play(p) => assert!(game.board().is_legal(p.x, p.y)),
            crate::core::Move::Pass => {}
            crate::core::Move::Resign => panic!("AI 不应认输"),
        }
    }
}
