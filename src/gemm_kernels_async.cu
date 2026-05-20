#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_pipeline_primitives.h>
#include <cstdio>
#include <torch/extension.h>

// ==================== Kernel 5: Register Tiling + cp.async Double Buffering ====================
#define BM 128
#define BN 128
#define BK 16  // Optimal: BK=8 too small for overlap, BK=32 exceeds SMEM budget with DB
#define TM 8
#define TN 8

__launch_bounds__(256, 2)
__global__ void gemm_db_async_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Ping-pong shared memory buffers (2 stages)
    __shared__ float As[2][BM * BK];
    __shared__ float Bs[2][BK * BN];

    float accum[TM][TN] = {0.0f};
    float reg_a[TM] = {0.0f};
    float reg_b[TN] = {0.0f};

    int block_r = blockIdx.y * BM;
    int block_c = blockIdx.x * BN;

    int thread_r = threadIdx.y * TM;
    int thread_c = threadIdx.x * TN;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int LOADS = BK / 8;  // float4 loads per thread per tile (1 for BK=8, 2 for BK=16, 4 for BK=32)

    int nTiles = (K + BK - 1) / BK;

    // ====== Synchronous preload of tile 0 to stage 0 ======
    for (int l = 0; l < LOADS; l++) {
        // Load A tile 0
        int a_row = tid / 2;
        int a_col = l * 8 + (tid % 2) * 4;
        int g_a_r = block_r + a_row;
        int g_a_c = a_col;  // tile 0, K offset = 0
        if (g_a_r < M && g_a_c < K) {
            float4 val = *reinterpret_cast<const float4*>(&A[g_a_r * K + g_a_c]);
            As[0][a_row * BK + a_col + 0] = val.x;
            As[0][a_row * BK + a_col + 1] = val.y;
            As[0][a_row * BK + a_col + 2] = val.z;
            As[0][a_row * BK + a_col + 3] = val.w;
        } else {
            As[0][a_row * BK + a_col + 0] = 0.0f;
            As[0][a_row * BK + a_col + 1] = 0.0f;
            As[0][a_row * BK + a_col + 2] = 0.0f;
            As[0][a_row * BK + a_col + 3] = 0.0f;
        }

        // Load B tile 0
        int b_row = l * 8 + tid / 32;
        int b_col = (tid % 32) * 4;
        int g_b_r = b_row;  // tile 0, K offset = 0
        int g_b_c = block_c + b_col;
        if (g_b_r < K && g_b_c < N) {
            float4 val = *reinterpret_cast<const float4*>(&B[g_b_r * N + g_b_c]);
            Bs[0][b_row * BN + b_col + 0] = val.x;
            Bs[0][b_row * BN + b_col + 1] = val.y;
            Bs[0][b_row * BN + b_col + 2] = val.z;
            Bs[0][b_row * BN + b_col + 3] = val.w;
        } else {
            Bs[0][b_row * BN + b_col + 0] = 0.0f;
            Bs[0][b_row * BN + b_col + 1] = 0.0f;
            Bs[0][b_row * BN + b_col + 2] = 0.0f;
            Bs[0][b_row * BN + b_col + 3] = 0.0f;
        }
    }
    __syncthreads();

    int stage = 0;

    // ====== Main loop: async-load next tile while computing current tile ======
    for (int tile = 1; tile < nTiles; tile++) {
        int next_stage = stage ^ 1;

        // Async load A and B tiles into next_stage
        for (int l = 0; l < LOADS; l++) {
            int a_row = tid / 2;
            int a_col = l * 8 + (tid % 2) * 4;
            int g_a_r = block_r + a_row;
            int g_a_c = tile * BK + a_col;
            if (g_a_r < M && g_a_c < K) {
                __pipeline_memcpy_async(&As[next_stage][a_row * BK + a_col],
                                         &A[g_a_r * K + g_a_c], 16);
            } else {
                As[next_stage][a_row * BK + a_col + 0] = 0.0f;
                As[next_stage][a_row * BK + a_col + 1] = 0.0f;
                As[next_stage][a_row * BK + a_col + 2] = 0.0f;
                As[next_stage][a_row * BK + a_col + 3] = 0.0f;
            }

            int b_row = l * 8 + tid / 32;
            int b_col = (tid % 32) * 4;
            int g_b_r = tile * BK + b_row;
            int g_b_c = block_c + b_col;
            if (g_b_r < K && g_b_c < N) {
                __pipeline_memcpy_async(&Bs[next_stage][b_row * BN + b_col],
                                         &B[g_b_r * N + g_b_c], 16);
            } else {
                Bs[next_stage][b_row * BN + b_col + 0] = 0.0f;
                Bs[next_stage][b_row * BN + b_col + 1] = 0.0f;
                Bs[next_stage][b_row * BN + b_col + 2] = 0.0f;
                Bs[next_stage][b_row * BN + b_col + 3] = 0.0f;
            }
        }
        __pipeline_commit();

        // Wait for current stage and compute
        __pipeline_wait_prior(0);
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[stage][(thread_r + i) * BK + k];
            }
            float4 vb0 = *reinterpret_cast<float4*>(&Bs[stage][k * BN + (thread_c + 0)]);
            float4 vb1 = *reinterpret_cast<float4*>(&Bs[stage][k * BN + (thread_c + 4)]);
            reg_b[0] = vb0.x; reg_b[1] = vb0.y; reg_b[2] = vb0.z; reg_b[3] = vb0.w;
            reg_b[4] = vb1.x; reg_b[5] = vb1.y; reg_b[6] = vb1.z; reg_b[7] = vb1.w;
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();

        stage = next_stage;
    }

    // ====== Compute the last tile ======
    for (int k = 0; k < BK; ++k) {
        for (int i = 0; i < TM; ++i) {
            reg_a[i] = As[stage][(thread_r + i) * BK + k];
        }
        float4 vb0 = *reinterpret_cast<float4*>(&Bs[stage][k * BN + (thread_c + 0)]);
        float4 vb1 = *reinterpret_cast<float4*>(&Bs[stage][k * BN + (thread_c + 4)]);
        reg_b[0] = vb0.x; reg_b[1] = vb0.y; reg_b[2] = vb0.z; reg_b[3] = vb0.w;
        reg_b[4] = vb1.x; reg_b[5] = vb1.y; reg_b[6] = vb1.z; reg_b[7] = vb1.w;
        for (int i = 0; i < TM; ++i) {
            for (int j = 0; j < TN; ++j) {
                accum[i][j] += reg_a[i] * reg_b[j];
            }
        }
    }
    __syncthreads();

    // ====== Write back ======
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int global_r = block_r + thread_r + i;
            int global_c = block_c + thread_c + j;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = accum[i][j];
            }
        }
    }
}

torch::Tensor matmul_db_async_forward(torch::Tensor a, torch::Tensor b) {
    a = a.contiguous().cuda();
    b = b.contiguous().cuda();
    int M = a.size(0); int K = a.size(1); int N = b.size(1);
    auto output = torch::empty({M, N}, a.options());

    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    gemm_db_async_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), M, N, K);
    return output;
}
