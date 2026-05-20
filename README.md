# ops_cuda_gemm

[English](README_EN.md)

手写 CUDA GEMM (SGEMM) 从零优化——8 个优化级别，7 个 kernel 函数，从朴素实现到 Tensor Core，单卡 RTX 3060 Ti。

**8.10 TFLOPS, 50.0% FP32 峰值**，消费级 Ampere 硬件。

## 性能 (4096x4096 FP32)

| Level | Kernel | 技术 | 时间 | TFLOPS | 加速比 |
|-------|--------|------|------|--------|--------|
| 1 | Naive | Global Memory | 128.3 ms | 1.07 | 1.0x |
| 2 | Shared | SMEM Tiling 32x32 | 104.4 ms | 1.32 | 1.2x |
| 3 | Float4 | + LDG.128 向量化加载 | 104.3 ms | 1.32 | 1.2x |
| 4 | RegTile | 寄存器分块 (64 累加器) | 17.7 ms | 7.78* | 7.3x |
| 5 | DB Async | + cp.async 双缓冲 | 17.2 ms | 8.00* | 7.5x |
| 6 | WMMA TC | TF32 Tensor Core | 17.3 ms | 7.95 | 7.4x |
| 7 | LDS.128+LB | SMEM 向量化 + launch bounds | — | +2~6% | — |
| 8 | WMMA+DB | Tensor Core + cp.async | 17.0 ms | **8.10** | 7.6x |
| — | **cuBLAS** | Tensor Core + SASS | 12.5 ms | 11.0 | 10.3x |

> *Level 4-5 当前值含 LDS.128（Level 7 优化已直接写入源码）。原始值：RegTile 7.19, DB Async 7.54。Level 7 无独立 kernel——是跨 kernel 微调 pass。

## 快速开始

```bash
# 激活 conda 环境
conda activate ainfra

# 构建
python setup.py build_ext --inplace

# 性能测试（全部规模、全部 kernel）
python bench/benchmark.py

# Nsight Compute 分析（需管理员权限 + GPU 性能计数器访问）
bench/run_ncu.bat
```

**环境要求**: Python 3.10+, PyTorch 2.5+ (CUDA), CUDA Toolkit 12.x, NVIDIA GPU SM 8.0+ (Ampere)。

## 项目结构

```
ops_cuda_gemm/
├── src/
│   ├── gemm_kernels.cu              # Level 1-4,7: Naive→RegTile, LDS.128
│   ├── gemm_kernels_async.cu        # Level 5,7: cp.async DB, LDS.128
│   ├── gemm_kernels_tc.cu           # Level 6: WMMA TF32 Tensor Core
│   ├── gemm_kernels_tc_async.cu     # Level 8: WMMA + cp.async
│   └── gemm_wrapper.cpp             # pybind11 桥接 (7 函数, 8 级别)
├── bench/
│   ├── benchmark.py                 # 性能测试 (8 种规模, CSV 输出)
│   ├── profile_kernel.py            # NCU 分析用最小脚本
│   └── run_ncu.bat                  # NCU 批量分析（时间戳输出）
├── docs/
│   └── optimization_strategy.md     # 优化全解 (14 章, 21 道面试题)
├── profiles/                        # NCU .ncu-rep 输出
├── results/                         # Benchmark CSV（时间戳）
├── setup.py                         # torch CUDAExtension 构建
└── README.md / README_EN.md
```

## 优化路径

```
1.07 TFLOPS ─ Naive 全局内存直接读写
     │
1.32 TFLOPS ─ Shared Memory 分块 (32×32 tiles)
     │
1.32 TFLOPS ─ Float4 LDG.128（瓶颈不在此 — Amdahl 定律）
     │
7.19 TFLOPS ─ 寄存器分块 (64 累加器, TM=TN=8) ← 转折点
     │          (现代码含 LDS.128: 7.78)
7.54 TFLOPS ─ cp.async 双缓冲（加载与计算开始重叠）
     │          (现代码含 LDS.128: 8.00)
     │  + LDS.128 + __launch_bounds__ ─ 跨 kernel 微调 (+2~6%)
     │
7.95 TFLOPS ─ WMMA TF32 Tensor Core（92% warp 空闲于 barrier）
     │
8.10 TFLOPS ─ WMMA + cp.async（CUDA Core 和 Tensor Core 首次持平）
```

> 8 个优化级别，7 个 kernel 函数。Level 7 (LDS.128) 是跨 kernel 优化 pass，直接应用于已有 kernel——未创建新函数。

## Nsight Compute 分析

| Kernel | SM Busy | IPC | No Eligible | Memory SOL | Compute SOL |
|--------|---------|-----|-------------|------------|-------------|
| Naive | 31.4% | 1.10 | 72.6% | 93.5% | 93.5% |
| Shared | 28.2% | 0.90 | 77.5% | 80.7% | 80.7% |
| Float4 | 26.3% | 0.83 | 79.2% | 74.8% | 74.8% |
| RegTile | 47.1% | 1.88 | 52.8% | 70.0% | 40.9% |
| DB Async | **49.3%** | **1.97** | 50.5% | 64.8% | 43.8% |
| WMMA TC | 19.3% | 0.32 | **91.9%** | 66.1% | 16.6% |
| WMMA+DB | 19.0% | 0.40 | 90.1% | 60.4% | 17.8% |

完整逐 kernel NCU 分析见 `docs/optimization_strategy.md` 第 13 章。

## 核心发现

- **CUDA Core 天花板**: 8.00 TFLOPS (49.4% FP32 峰值)。IPC 1.97 触及 Ampere 双发射上限。剩余 50% SM 空闲来自 Shared Memory 延迟 + barrier 同步。
- **Tensor Core 瓶颈**: 不是算力——是 `__syncthreads()`。92% warp 空闲。WMMA `load_matrix_sync` + barrier 串行化让 Tensor Core 饿死。
- **cuBLAS 差距 (11.0 vs 8.10)**: cuBLAS 使用手写 SASS 汇编、3-4 stage cp.async pipeline、XOR swizzle、异步 barrier——这些都无法在纯手写 CUDA C++ 中实现。

## 文档

`docs/optimization_strategy.md` — 14 章，涵盖：
- GPU 内存层次结构、Roofline 模型
- 8 个优化级别的完整代码解读
- 21 道面试题 + 详细中文话术回答
- 7 kernel 全 NCU 分析
- cuBLAS 差距深度剖析

## License

MIT
