"""Minimal script for NCU profiling — runs one kernel at one size."""
import os; os.add_dll_directory(r"D:\anaconda\envs\ainfra\lib\site-packages\torch\lib")
import torch, sys

kernel_name = sys.argv[1]
M = K = N = 1024  # Standard profiling size

import gemm_ops
kernel_map = {
    "matmul_kernel":                  gemm_ops.matmul_naive,
    "matmul_shared_kernel":           gemm_ops.matmul_shared,
    "matmul_shared_float4_kernel":    gemm_ops.matmul_float4,
    "matmul_register_tiling_kernel":  gemm_ops.matmul_register,
    "gemm_db_async_kernel":           gemm_ops.matmul_db_async,
    "gemm_wmma_kernel":               gemm_ops.matmul_wmma,
    "gemm_wmma_async_kernel":         gemm_ops.matmul_wmma_async,
}

func = kernel_map[kernel_name]
q = torch.randn(M, K, device='cuda', dtype=torch.float32)
k = torch.randn(K, N, device='cuda', dtype=torch.float32)

# Warmup
for _ in range(5):
    func(q, k)
torch.cuda.synchronize()

# Profiling iterations — NCU will capture the first one after -s skip
for _ in range(10):
    func(q, k)
torch.cuda.synchronize()
