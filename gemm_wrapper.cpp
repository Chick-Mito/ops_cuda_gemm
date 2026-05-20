#include <torch/extension.h>

// Forward declarations
torch::Tensor matmul_naive_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor matmul_shared_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor matmul_float4_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor matmul_register_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor matmul_db_async_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor matmul_wmma_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor matmul_wmma_async_forward(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_naive",       &matmul_naive_forward,       "Naive Matmul (Global Memory)");
    m.def("matmul_shared",      &matmul_shared_forward,      "Shared Memory Tiling Matmul");
    m.def("matmul_float4",      &matmul_float4_forward,      "Float4 Vectorized + Shared Memory Matmul");
    m.def("matmul_register",    &matmul_register_forward,    "Register Tiling + Float4 + Shared Memory Matmul");
    m.def("matmul_db_async",    &matmul_db_async_forward,    "Register Tiling + cp.async Double Buffering");
    m.def("matmul_wmma",        &matmul_wmma_forward,        "WMMA TF32 Tensor Core GEMM");
    m.def("matmul_wmma_async",  &matmul_wmma_async_forward,  "WMMA TF32 + cp.async Double Buffering");
}
