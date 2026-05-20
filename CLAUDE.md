# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Conda env**: `D:\anaconda\envs\ainfra` (Python 3.10.20, PyTorch 2.5.1+cu121)
- **GPU**: NVIDIA GeForce RTX 3060 Ti (8GB, sm_86, Ampere)
- **CUDA Toolkit**: v12.4 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`)
- **Nsight Compute**: 2024.1.1 (`C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.1\ncu.bat`)

## Commands

```bash
# Build the CUDA extension (run from project root)
conda activate ainfra
python setup.py build_ext --inplace

# Run benchmarks
python benchmark.py

# Profile all kernels with Nsight Compute (requires admin privileges)
run_ncu.bat

# Quick smoke test after build
python -c "import os; os.add_dll_directory(r'D:\anaconda\envs\ainfra\lib\site-packages\torch\lib'); import gemm_ops; print([f for f in dir(gemm_ops) if not f.startswith('_')])"
```

**Important**: After `build_ext --inplace`, copy the `.pyd` from `build/lib.win-amd64-cpython-310/` to the project root if it wasn't auto-copied. The `.pyd` must be importable as `import gemm_ops`.

**DLL Path Issue**: `gemm_ops.pyd` links against `c10.dll`, `torch_cpu.dll`, `torch_python.dll`, `cudart64_12.dll` which live in `D:\anaconda\envs\ainfra\lib\site-packages\torch\lib\`. Before importing `gemm_ops`, call `os.add_dll_directory(r"D:\anaconda\envs\ainfra\lib\site-packages\torch\lib")`. `benchmark.py` already does this.

## Architecture

### Kernel Hierarchy (4 levels, all in `gemm_kernels.cu`)

| Level | Kernel | SMEM | Load Tech | Per-thread accum | TFLOPS (4096²) |
|-------|--------|------|-----------|-------------------|----------------|
| 1 | `matmul_kernel` | None | Scalar LDG | 1 | 1.0 |
| 2 | `matmul_shared_kernel` | 32×32 (8 KB) | Scalar LDG | 1 | 1.3 |
| 3 | `matmul_shared_float4_kernel` | 32×32 (8 KB) | float4 LDG (128-bit) | 1 | 1.3 |
| 4 | `matmul_register_tiling_kernel` | 128×8+8×128 (8 KB) | float4 LDG (128-bit) | 8×8=64 | 7.3 |

Each kernel has a C++ host-side bridge function (e.g. `matmul_register_forward`) in the same `.cu` file that handles tensor→CUDA conversion, grid/block config, and launch.

### Build System

- `setup.py` — `torch.utils.cpp_extension.CUDAExtension`, compiles `.cu` with nvcc (`-O3 -gencode=arch=compute_86,code=sm_86`), links against torch/cudart
- `gemm_wrapper.cpp` — pybind11 module (`gemm_ops`) exposing 4 matmul functions to Python
- Output: `gemm_ops.cp310-win_amd64.pyd`

### Python Layer

`benchmark.py` uses `torch.cuda.Event` for GPU-precise timing (not `time.time()`):
1. 10 warmup iterations
2. `start_event.record()` → 50 timing iterations → `end_event.record()`
3. `torch.cuda.synchronize()` → `start_event.elapsed_time(end_event) / 50`
4. Validates correctness via `(out - torch.mm(q,k)).abs().max() < 0.01`

## Constraints & Gotchas

- **float4 alignment**: float4 loads require K and N to be multiples of 4 (16-byte aligned addresses). Kernels 3 & 4 are auto-skipped in benchmark when this isn't met.
- **nvcc target**: Compilation is hardcoded to `sm_86` — only runs on Ampere GPUs (RTX 3060 Ti or similar).
- **TC not used**: The 4 current kernels use CUDA Cores only (no Tensor Core / WMMA).
- **cuBLAS baseline**: `torch.mm()` achieves ~11 TFLOPS (68% peak) on 4096² FP32.
- **`os.add_dll_directory` is required** before `import gemm_ops` when running outside a fully activated conda prompt.

## Reference Code

- `C:\Users\Citrus\Desktop\float4 2.0\` — Original project with 4 GEMM + 3 softmax kernels, NCU profiles (`.ncu-rep`), and optimization notes
- Build approach identical to this project: `torch.utils.cpp_extension.CUDAExtension` + `pybind11`
