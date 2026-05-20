#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <torch/extension.h>

#define BLOCK_SIZE 32

// ==================== Kernel 1: Naive Matmul ====================
__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N) {
        float sum = 0.0f;
        for(int i=0; i<K; i++) {
            sum += A[row*K + i] * B[i*N + col];
        }
        C[row * N + col] = sum;
    }
}

// ==================== Kernel 2: Shared Memory Matmul ====================
__global__ void matmul_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
){
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for(int tile = 0; tile<(K-1+BLOCK_SIZE)/BLOCK_SIZE; ++tile) {
        // Load A tile
        int a_row = row;
        int a_col = tile * BLOCK_SIZE + threadIdx.x;
        if(a_row<M && a_col<K) {
            As[threadIdx.y][threadIdx.x] = A[a_row*K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile
        int b_row = tile * BLOCK_SIZE + threadIdx.y;
        int b_col = col;
        if(b_row<K && b_col<N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row*N + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for(int k=0; k<BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row<M && col<N) {
        C[row*N + col] = sum;
    }
}

// ==================== Kernel 3: Float4 Vectorized + Shared Memory ====================
__global__ void matmul_shared_float4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
){
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for(int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Float4 load for A tile (only first 256 threads)
        int tile_a_r = blockIdx.y * BLOCK_SIZE;
        int tile_a_c = tile * BLOCK_SIZE;

        if (tid < 256) {
            int load_r = tid / 8;
            int load_c = (tid % 8) * 4;

            int global_r = tile_a_r + load_r;
            int global_c = tile_a_c + load_c;

            if (global_r < M && global_c < K) {
                const float4* A_ptr = reinterpret_cast<const float4*>(&A[global_r * K + global_c]);
                float4 a_vec = *A_ptr;

                As[load_r][load_c + 0] = a_vec.x;
                As[load_r][load_c + 1] = a_vec.y;
                As[load_r][load_c + 2] = a_vec.z;
                As[load_r][load_c + 3] = a_vec.w;
            } else {
                As[load_r][load_c + 0] = 0.0f;
                As[load_r][load_c + 1] = 0.0f;
                As[load_r][load_c + 2] = 0.0f;
                As[load_r][load_c + 3] = 0.0f;
            }
        }

        // Float4 load for B tile (only first 256 threads)
        int tile_b_r = tile * BLOCK_SIZE;
        int tile_b_c = blockIdx.x * BLOCK_SIZE;

        if (tid < 256) {
            int load_r = tid / 8;
            int load_c = (tid % 8) * 4;

            int global_r = tile_b_r + load_r;
            int global_c = tile_b_c + load_c;

            if (global_r < K && global_c < N) {
                const float4* B_ptr = reinterpret_cast<const float4*>(&B[global_r * N + global_c]);
                float4 b_vec = *B_ptr;

                Bs[load_r][load_c + 0] = b_vec.x;
                Bs[load_r][load_c + 1] = b_vec.y;
                Bs[load_r][load_c + 2] = b_vec.z;
                Bs[load_r][load_c + 3] = b_vec.w;
            } else {
                Bs[load_r][load_c + 0] = 0.0f;
                Bs[load_r][load_c + 1] = 0.0f;
                Bs[load_r][load_c + 2] = 0.0f;
                Bs[load_r][load_c + 3] = 0.0f;
            }
        }

        __syncthreads();

        for(int k=0; k<BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ==================== Kernel 4: Register Tiling + Float4 + Shared Memory ====================
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
__launch_bounds__(256, 2)
__global__ void matmul_register_tiling_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[BM * BK]; // 128 x 8
    __shared__ float Bs[BK * BN]; // 8 x 128

    float accum[TM][TN] = {0.0f};
    float reg_a[TM] = {0.0f};
    float reg_b[TN] = {0.0f};

    int block_r = blockIdx.y * BM;
    int block_c = blockIdx.x * BN;

    int thread_r = threadIdx.y * TM;
    int thread_c = threadIdx.x * TN;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int a_row_load = tid / 2;         // 0 ~ 127
    int a_col_load = (tid % 2) * 4;   // 0 or 4

    int b_row_load = tid / 32;        // 0 ~ 7
    int b_col_load = (tid % 32) * 4;  // 0, 4, 8 ... 124

    for (int tile = 0; tile < (K + BK - 1) / BK; ++tile) {
        // Float4 cooperative load A tile
        int global_a_r = block_r + a_row_load;
        int global_a_c = tile * BK + a_col_load;
        if (global_a_r < M && global_a_c < K) {
            float4 val = *reinterpret_cast<const float4*>(&A[global_a_r * K + global_a_c]);
            As[a_row_load * BK + a_col_load + 0] = val.x;
            As[a_row_load * BK + a_col_load + 1] = val.y;
            As[a_row_load * BK + a_col_load + 2] = val.z;
            As[a_row_load * BK + a_col_load + 3] = val.w;
        } else {
            As[a_row_load * BK + a_col_load + 0] = 0.0f;
            As[a_row_load * BK + a_col_load + 1] = 0.0f;
            As[a_row_load * BK + a_col_load + 2] = 0.0f;
            As[a_row_load * BK + a_col_load + 3] = 0.0f;
        }

        // Float4 cooperative load B tile
        int global_b_r = tile * BK + b_row_load;
        int global_b_c = block_c + b_col_load;
        if (global_b_r < K && global_b_c < N) {
            float4 val = *reinterpret_cast<const float4*>(&B[global_b_r * N + global_b_c]);
            Bs[b_row_load * BN + b_col_load + 0] = val.x;
            Bs[b_row_load * BN + b_col_load + 1] = val.y;
            Bs[b_row_load * BN + b_col_load + 2] = val.z;
            Bs[b_row_load * BN + b_col_load + 3] = val.w;
        } else {
            Bs[b_row_load * BN + b_col_load + 0] = 0.0f;
            Bs[b_row_load * BN + b_col_load + 1] = 0.0f;
            Bs[b_row_load * BN + b_col_load + 2] = 0.0f;
            Bs[b_row_load * BN + b_col_load + 3] = 0.0f;
        }
        __syncthreads();

        // Register-level compute
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[(thread_r + i) * BK + k];
            }
            // LDS.128: load 8 B values in 2 float4 instead of 8 scalar
            float4 vb0 = *reinterpret_cast<float4*>(&Bs[k * BN + (thread_c + 0)]);
            float4 vb1 = *reinterpret_cast<float4*>(&Bs[k * BN + (thread_c + 4)]);
            reg_b[0] = vb0.x; reg_b[1] = vb0.y; reg_b[2] = vb0.z; reg_b[3] = vb0.w;
            reg_b[4] = vb1.x; reg_b[5] = vb1.y; reg_b[6] = vb1.z; reg_b[7] = vb1.w;

            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    // Write back 8x8 block to global memory
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

// ==================== Bridge Functions ====================
torch::Tensor matmul_naive_forward(torch::Tensor a, torch::Tensor b) {
    a = a.contiguous().cuda();
    b = b.contiguous().cuda();
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
    auto output = torch::empty({M, N}, a.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), M, N, K);
    return output;
}

torch::Tensor matmul_shared_forward(torch::Tensor a, torch::Tensor b) {
    a = a.contiguous().cuda();
    b = b.contiguous().cuda();
    int M = a.size(0); int K = a.size(1); int N = b.size(1);
    auto output = torch::empty({M, N}, a.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_shared_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), M, N, K);
    return output;
}

torch::Tensor matmul_float4_forward(torch::Tensor a, torch::Tensor b) {
    a = a.contiguous().cuda();
    b = b.contiguous().cuda();
    int M = a.size(0); int K = a.size(1); int N = b.size(1);
    auto output = torch::empty({M, N}, a.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_shared_float4_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), M, N, K);
    return output;
}

torch::Tensor matmul_register_forward(torch::Tensor a, torch::Tensor b) {
    a = a.contiguous().cuda();
    b = b.contiguous().cuda();
    int M = a.size(0); int K = a.size(1); int N = b.size(1);
    auto output = torch::empty({M, N}, a.options());

    dim3 block(16, 16);
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);

    matmul_register_tiling_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), M, N, K);
    return output;
}
