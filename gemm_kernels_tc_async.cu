#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_pipeline_primitives.h>
#include <cstdio>
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;

#define TC_BM 128
#define TC_BN 128
#define TC_BK 16
#define WMMA_K 8

__launch_bounds__(256, 2)
__global__ void gemm_wmma_async_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Ping-pong shared memory for double buffering
    __shared__ float As[2][TC_BM][TC_BK];
    __shared__ float BsT[2][TC_BN][TC_BK];

    // Per-warp accumulators (must persist across tiles)
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[4][2];
    for (int tr = 0; tr < 4; tr++)
        for (int tc = 0; tc < 2; tc++)
            wmma::fill_fragment(c_frag[tr][tc], 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_frag;

    int warpRow = threadIdx.y;
    int warpCol = threadIdx.x / 32;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Load indices for cooperative loading (same as sync WMMA kernel)
    int a_row = tid / 2;
    int a_col_lo = (tid % 2) * 4;
    int b_k_lo  = tid / 32;
    int b_n_lo  = (tid % 32) * 4;

    int block_r = blockIdx.y * TC_BM;
    int block_c = blockIdx.x * TC_BN;
    int nTiles = (K + TC_BK - 1) / TC_BK;
    const int SUBSTEPS = TC_BK / WMMA_K;

    // ====== Stage 0: synchronous preload ======
    for (int l = 0; l < SUBSTEPS; l++) {
        int a_col = l * WMMA_K + a_col_lo;
        int ga_r = block_r + a_row;
        int ga_c = a_col;
        if (ga_r < M && ga_c < K) {
            float4 v = *reinterpret_cast<const float4*>(&A[ga_r * K + ga_c]);
            As[0][a_row][a_col+0]=v.x; As[0][a_row][a_col+1]=v.y;
            As[0][a_row][a_col+2]=v.z; As[0][a_row][a_col+3]=v.w;
        } else {
            As[0][a_row][a_col+0]=0.f; As[0][a_row][a_col+1]=0.f;
            As[0][a_row][a_col+2]=0.f; As[0][a_row][a_col+3]=0.f;
        }
    }
    for (int l = 0; l < SUBSTEPS; l++) {
        int b_k = l * WMMA_K + b_k_lo;
        int gb_r = b_k;
        int gb_c = block_c + b_n_lo;
        if (gb_r < K && gb_c < N) {
            float4 v = *reinterpret_cast<const float4*>(&B[gb_r * N + gb_c]);
            BsT[0][b_n_lo+0][b_k]=v.x; BsT[0][b_n_lo+1][b_k]=v.y;
            BsT[0][b_n_lo+2][b_k]=v.z; BsT[0][b_n_lo+3][b_k]=v.w;
        } else {
            BsT[0][b_n_lo+0][b_k]=0.f; BsT[0][b_n_lo+1][b_k]=0.f;
            BsT[0][b_n_lo+2][b_k]=0.f; BsT[0][b_n_lo+3][b_k]=0.f;
        }
    }
    __syncthreads();

    int stage = 0;

    // ====== Main loop: async-load next while computing current ======
    for (int tile = 1; tile < nTiles; tile++) {
        int next = stage ^ 1;

        // --- Async load next tile ---
        for (int l = 0; l < SUBSTEPS; l++) {
            int a_col = l * WMMA_K + a_col_lo;
            int ga_r = block_r + a_row;
            int ga_c = tile * TC_BK + a_col;
            if (ga_r < M && ga_c < K) {
                __pipeline_memcpy_async(&As[next][a_row][a_col],
                    &A[ga_r * K + ga_c], 16);
            } else {
                As[next][a_row][a_col+0]=0.f; As[next][a_row][a_col+1]=0.f;
                As[next][a_row][a_col+2]=0.f; As[next][a_row][a_col+3]=0.f;
            }
        }
        for (int l = 0; l < SUBSTEPS; l++) {
            int b_k = l * WMMA_K + b_k_lo;
            int gb_r = tile * TC_BK + b_k;
            int gb_c = block_c + b_n_lo;
            if (gb_r < K && gb_c < N) {
                float4 v = *reinterpret_cast<const float4*>(&B[gb_r * N + gb_c]);
                BsT[next][b_n_lo+0][b_k]=v.x; BsT[next][b_n_lo+1][b_k]=v.y;
                BsT[next][b_n_lo+2][b_k]=v.z; BsT[next][b_n_lo+3][b_k]=v.w;
            } else {
                BsT[next][b_n_lo+0][b_k]=0.f; BsT[next][b_n_lo+1][b_k]=0.f;
                BsT[next][b_n_lo+2][b_k]=0.f; BsT[next][b_n_lo+3][b_k]=0.f;
            }
        }
        __pipeline_commit();

        // --- Wait for current stage + WMMA compute ---
        __pipeline_wait_prior(0);
        __syncthreads();

        for (int ks = 0; ks < SUBSTEPS; ks++) {
            for (int tr = 0; tr < 4; tr++) {
                for (int tc = 0; tc < 2; tc++) {
                    int mBase = warpRow * 64 + tr * 16;
                    int nBase = warpCol * 32 + tc * 16;
                    wmma::load_matrix_sync(a_frag, &As[stage][mBase][ks * WMMA_K], TC_BK);
                    wmma::load_matrix_sync(b_frag, &BsT[stage][nBase][ks * WMMA_K], TC_BK);
                    wmma::mma_sync(c_frag[tr][tc], a_frag, b_frag, c_frag[tr][tc]);
                }
            }
        }
        __syncthreads();

        stage = next;
    }

    // ====== Compute last tile ======
    for (int ks = 0; ks < SUBSTEPS; ks++) {
        for (int tr = 0; tr < 4; tr++) {
            for (int tc = 0; tc < 2; tc++) {
                int mBase = warpRow * 64 + tr * 16;
                int nBase = warpCol * 32 + tc * 16;
                wmma::load_matrix_sync(a_frag, &As[stage][mBase][ks * WMMA_K], TC_BK);
                wmma::load_matrix_sync(b_frag, &BsT[stage][nBase][ks * WMMA_K], TC_BK);
                wmma::mma_sync(c_frag[tr][tc], a_frag, b_frag, c_frag[tr][tc]);
            }
        }
    }
    __syncthreads();

    // ====== Store ======
    for (int tr = 0; tr < 4; tr++) {
        for (int tc = 0; tc < 2; tc++) {
            int out_r = block_r + warpRow * 64 + tr * 16;
            int out_c = block_c + warpCol * 32 + tc * 16;
            if (out_r < M && out_c < N) {
                wmma::store_matrix_sync(&C[out_r * N + out_c],
                    c_frag[tr][tc], N, wmma::mem_row_major);
            }
        }
    }
}

torch::Tensor matmul_wmma_async_forward(torch::Tensor a, torch::Tensor b) {
    a = a.contiguous().cuda();
    b = b.contiguous().cuda();
    int M = a.size(0); int K = a.size(1); int N = b.size(1);
    auto output = torch::empty({M, N}, a.options());

    dim3 block(128, 2);
    dim3 grid((N + TC_BN - 1) / TC_BN, (M + TC_BM - 1) / TC_BM);

    gemm_wmma_async_kernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), M, N, K);
    return output;
}
