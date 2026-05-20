# ops_cuda_gemm

[中文版](README_CN.md)

Hand-rolled CUDA GEMM (SGEMM) optimization from scratch — 8 optimization levels, 7 kernel functions, from naive to Tensor Core, on a single RTX 3060 Ti.

**8.10 TFLOPS, 50.0% FP32 peak** on consumer Ampere hardware.

## Performance (4096x4096 FP32)

| Level | Kernel | Technology | Time | TFLOPS | Speedup |
|-------|--------|-----------|------|--------|---------|
| 1 | Naive | Global Memory | 128.3 ms | 1.07 | 1.0x |
| 2 | Shared | SMEM Tiling 32x32 | 104.4 ms | 1.32 | 1.2x |
| 3 | Float4 | + LDG.128 vectorized load | 104.3 ms | 1.32 | 1.2x |
| 4 | RegTile | Register Tiling (64 accum) | 17.7 ms | 7.78* | 7.3x |
| 5 | DB Async | + cp.async Double Buffering | 17.2 ms | 8.00* | 7.5x |
| 6 | WMMA TC | TF32 Tensor Core | 17.3 ms | 7.95 | 7.4x |
| 7 | LDS.128+LB | SMEM vectorized load + launch bounds | — | +2~6% | — |
| 8 | WMMA+DB | Tensor Core + cp.async | 17.0 ms | **8.10** | 7.6x |
| — | **cuBLAS** | Tensor Core + SASS | 12.5 ms | 11.0 | 10.3x |

> *Level 4-5 当前值含 LDS.128（Level 7 优化已直接写入源码）。原始值：RegTile 7.19, DB Async 7.54。Level 7 无独立 kernel——是跨 kernel 微调 pass。Level 1-4→7 在 `src/gemm_kernels.cu`，Level 5→7 在 `src/gemm_kernels_async.cu`，Level 6 在 `src/gemm_kernels_tc.cu`，Level 8 在 `src/gemm_kernels_tc_async.cu`。

## Quick Start

```bash
# Activate conda environment
conda activate ainfra

# Build
python setup.py build_ext --inplace

# Benchmark (all sizes, all kernels)
python benchmark.py

# Nsight Compute profiling (requires admin + GPU perf counter access)
run_ncu.bat
```

**Requirements**: Python 3.10+, PyTorch 2.5+ with CUDA, CUDA Toolkit 12.x, NVIDIA GPU SM 8.0+ (Ampere).

## Project Structure

```
ops_cuda_gemm/
├── src/
│   ├── gemm_kernels.cu              # Level 1-4,7: Naive→RegTile, LDS.128
│   ├── gemm_kernels_async.cu        # Level 5,7: cp.async DB, LDS.128
│   ├── gemm_kernels_tc.cu           # Level 6: WMMA TF32 Tensor Core
│   ├── gemm_kernels_tc_async.cu     # Level 8: WMMA + cp.async
│   └── gemm_wrapper.cpp             # pybind11 bridge (7 functions, 8 levels)
├── bench/
│   ├── benchmark.py                 # Performance test suite (8 sizes, CSV output)
│   ├── profile_kernel.py            # Minimal script for NCU profiling
│   └── run_ncu.bat                  # Batch NCU profiler (timestamped outputs)
├── docs/
│   └── optimization_strategy.md     # Full optimization doc (14 chapters, 21 interview Q&A)
├── profiles/                        # NCU .ncu-rep outputs
├── results/                         # Benchmark CSV outputs (timestamped)
├── setup.py                         # torch CUDAExtension build
└── README.md
```

## Optimization Journey

```
1.07 TFLOPS ─ Naive global memory
     │
1.32 TFLOPS ─ Shared Memory Tiling (32×32 tiles)
     │
1.32 TFLOPS ─ Float4 LDG.128 (bottleneck shifted — Amdahl's Law)
     │
7.19 TFLOPS ─ Register Tiling (64 accum/thread, TM=TN=8) ← breakthrough
     │          (current code with LDS.128: 7.78)
7.54 TFLOPS ─ cp.async Double Buffering (load + compute overlap)
     │          (current code with LDS.128: 8.00)
     │  + LDS.128 + __launch_bounds__ ─ cross-cutting micro-opt (+2~6%)
     │
7.95 TFLOPS ─ WMMA TF32 Tensor Core (92% warp stall on barrier)
     │
8.10 TFLOPS ─ WMMA + cp.async (CUDA Core and Tensor Core finally converge)
```

> 8 optimization levels, 7 kernel functions. Level 7 (LDS.128) is a cross-cutting pass applied to existing kernels — no new kernel created.

## Nsight Compute Analysis

| Kernel | SM Busy | IPC | No Eligible | Memory SOL | Compute SOL |
|--------|---------|-----|-------------|------------|-------------|
| Naive | 31.4% | 1.10 | 72.6% | 93.5% | 93.5% |
| Shared | 28.2% | 0.90 | 77.5% | 80.7% | 80.7% |
| Float4 | 26.3% | 0.83 | 79.2% | 74.8% | 74.8% |
| RegTile | 47.1% | 1.88 | 52.8% | 70.0% | 40.9% |
| DB Async | **49.3%** | **1.97** | 50.5% | 64.8% | 43.8% |
| WMMA TC | 19.3% | 0.32 | **91.9%** | 66.1% | 16.6% |
| WMMA+DB | 19.0% | 0.40 | 90.1% | 60.4% | 17.8% |

Full per-kernel NCU breakdown in `docs/optimization_strategy.md` Chapter 13.

## Key Findings

- **CUDA Core ceiling**: 8.00 TFLOPS (49.4% FP32 peak). IPC 1.97 hits Ampere dual-issue limit. Remaining 50% SM idle is shared memory latency + barrier sync.
- **Tensor Core bottleneck**: Not compute — it's `__syncthreads()`. 92% warp idle. WMMA `load_matrix_sync` + barrier serialization starves the Tensor Cores.
- **cuBLAS gap (11.0 vs 8.10)**: cuBLAS uses hand-tuned SASS assembly, 3-4 stage cp.async pipeline, XOR swizzle, and async barriers — none of which are practical to replicate in hand-written CUDA C++.

## Documentation

`docs/optimization_strategy.md` — 14 chapters covering:
- GPU memory hierarchy, Roofline model
- 8 optimization levels with full code walkthrough
- 21 interview-style questions with detailed Chinese answer scripts
- NCU profiling analysis across all kernels
- cuBLAS gap deep-dive

## License

MIT
