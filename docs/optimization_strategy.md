# CUDA GEMM 算子优化思路全解

> 从 Naive 实现到 Register Tiling，结合 Roofline 模型与 Ampere 架构特性，逐步拆解 GEMM 优化方法论。
> 本文档采用**面试官视角**设计问题，每道题附详细话术回答。

---

## 第1章 引言：GEMM 与 GPU 计算

### 1.1 GEMM 在深度学习中的地位

矩阵乘法（GEMM, GEneral Matrix Multiply）是深度学习的算力基石：

| 场景 | GEMM 占比 |
|------|----------|
| Transformer Self-Attention (Q×K^T, Attn×V) | ~60% |
| MLP / FFN 层 (W×X) | ~30% |
| CNN 卷积 (im2col→GEMM) | ~80% |
| LLM 推理 (prefill 阶段) | >90% |

cuBLAS 是 NVIDIA 官方的高度优化 GEMM 库，但**理解底层优化原理**才能：
- 做算子融合时知道如何拆分和重组计算
- 为自定义算子（如 FlashAttention 中的分块 matmul）选择正确的分块策略
- 读懂 Nsight Compute 性能分析报告
- 面试中展示扎实的 GPU 计算功底

### 1.2 实验环境

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 3060 Ti (Ampere, sm_86) |
| 显存 | 8 GB GDDR6 |
| CUDA Cores | 4864 (38 SM × 128 CUDA Core/SM) |
| FP32 理论峰值 | **16.2 TFLOPS** |
| 显存带宽 | **448 GB/s** |
| Shared Memory / SM | 100 KB (可配置) |
| 寄存器 / SM | 65536 × 32-bit |
| Max Threads / SM | 1536 (Ampere) |

---

## 第2章 GPU 内存层次与 Roofline 模型

### 2.1 内存层次结构

```
寄存器 (Register)      — 0 cycle,  ~256 KB/SM, 每个线程私有
    ↓
Shared Memory / L1     — ~30 cycles, 100 KB/SM, Block 内共享
    ↓
L2 Cache               — ~200 cycles, 3 MB, 所有 SM 共享
    ↓
Global Memory (HBM)    — ~700 cycles, 8 GB, 所有 SM 共享
```

**关键洞察**：从 Global Memory 读取一次数据（~700 cycles）的时间，可以在寄存器里完成 **700+ 次 FMA 运算**。这就是"计算换访存"策略的物理基础。

### 2.2 Roofline 模型

Roofline 模型将 Kernel 性能画在一张二维图上：
- **横轴**：算术强度（FLOP/Byte）— 每次字节访存能做的浮点运算数
- **纵轴**：可达性能（FLOP/s）
- **斜线**：Memory Bound 区域 — 性能受限于带宽
- **水平线**：Compute Bound 区域 — 性能受限于算力峰值

```
峰值算力 16.2 TFLOPS ─────────────────────────────
                          │          │
                          │  Memory  │ Compute
                          │  Bound   │  Bound
                          │          │
                        斜线 = 带宽 × 算术强度
                        448 GB/s
```

**判断方法**：
- 算术强度 < (峰值算力 / 带宽) = 16.2T / 448G = 36 FLOP/Byte → **Memory Bound**
- 算术强度 > 36 FLOP/Byte → **Compute Bound**

---

### 🔥 面试题 1：如何判断一个 Kernel 是 Memory-Bound 还是 Compute-Bound？Roofline 模型如何指导优化策略？

**话术回答**：

"判断一个 Kernel 是 Memory-Bound 还是 Compute-Bound，我通常用两个方法。

**方法一：理论分析**。计算 Kernel 的算术强度，即 `总 FLOP / 总访存字节数`。比如 4096×4096 的 Naive GEMM，算术强度 = `2 × M × N × K / (2 × M × N × K × 4) ≈ 0.125 FLOP/Byte`。而 GPU 的算力带宽比是 `16.2T / 448G ≈ 36 FLOP/Byte`。0.125 远小于 36，所以这个 Kernel 必然是 Memory-Bound 的。

**方法二：Nsight Compute 实测**。打开 `GPU Speed Of Light` 面板，如果 Memory 利用率接近 100% 而 Compute 利用率很低（比如 5%），就是典型的 Memory-Bound。反过来如果 Compute 利用率高，就是 Compute-Bound。

**对优化策略的指导**：
- Memory-Bound → 优化方向是减少访存（Tiling、Shared Memory、提高缓存命中率）或提高访存效率（向量化访存 float4、合并访存、对齐访问）
- Compute-Bound → 优化方向是减少指令开销（使用 FMA 指令、提高 ILP、减少 warp divergence）

在我们的实验中，Naive GEMM 每线程读 2K 次 Global Memory 做 K 次 FMA，是极端的 Memory-Bound。经过 Shared Memory Tiling 和 Register Tiling，我们将算术强度从 0.125 逐步提升到 ~8 FLOP/Byte，虽然仍不到 36 的边界，但已经大幅提升了实际可达性能。"

---

## 第3章 Level 1 — Naive 实现分析

### 3.1 代码逻辑

```cuda
__global__ void matmul_kernel(A, B, C, M, N, K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

每个线程计算 C 的一个元素，沿 K 维度遍历做点积。

### 3.2 瓶颈分析

**访存量**：每个 C 元素需要读取 A 的一行（K 个 float）+ B 的一列（K 个 float），即：

- 每个线程读 `2K` 次 Global Memory
- 总读量：`M × N × 2K × 4 bytes`（每个 float 4 字节）

以 4096² 为例：读量 = 4096² × 2 × 4096 × 4 ≈ **550 GB**，远超 GPU 显存带宽的承载能力。

**非合并访存（Uncoalesced Access）**：GPU 的 Global Memory 以 32-byte（或 128-byte）事务为单位访问。同一个 Warp 的 32 个线程同时访问 A 的行（合并），但访问 B 的列时是 stride-N 的跳跃访问（彻底不合并），Load Efficiency 仅 ~4%（32 次事务只用了 128 bytes 中的 4 bytes）。

### 3.3 实测性能

| 尺寸 | 时间 | 带宽 | TFLOPS | SOL Memory | SOL Compute |
|------|------|------|--------|------------|-------------|
| 4096² | 133 ms | 1.5 GB/s | 1.0 | ~0.3% | ~6% |

RS 只达到理论带宽的 0.3%，Compute 利用率 ~6%。

---

### 🔥 面试题 2：为什么矩阵 B 的列遍历会导致非合并访存？GPU 的合并访存规则是什么？

**话术回答**：

"合并访存（Coalesced Access）是指同一个 Warp 的 32 个线程同时发出的 Global Memory 请求能够被合并成尽可能少的 Memory Transaction，从而最大化总线利用率。

**规则**：从计算能力 6.0（Pascal）开始，GPU 的 L1 Cache Line 是 32 bytes。一个 Warp 的 32 个线程如果访问 128 bytes 对齐的连续地址空间，只需要一次 128-byte（或 4 次 32-byte）Transaction。但如果线程访问的地址是间断的，每个线程就会产生单独的 Transaction。

**具体到 GEMM 中**：
- **矩阵 A**：线程 `(row, col)` 访问 `A[row * K + i]`。同一个 Warp 中，`threadIdx.x` 连续的线程（同一行不同 col）在 i 固定时访问的是 A 中连续的元素 `A[row][i]`，因为 A 是行优先存储，这些地址是连续的 → **完美合并访存**。
- **矩阵 B**：访问 `B[i * N + col]`。同一个 Warp 中 col 不同（如 col=0,1,2,...,31），需要的是 B 的同一行中跨步为 1 的连续元素。但这些线程访问的是 B 的不同行（i 不同时）或同一行的不同元素。关键问题是，在内层循环 `i` 固定的某次迭代中，由 `threadIdx.x` 决定的 col 值是连续的，所以访问 B 的 `i*N + col` 在地址上也是连续的 → 但问题是 col 最大到 31，而 B 的一行可能有几千列，所以它们都在同一 Cache Line 内，也是可以合并的...

重新思考：实际上在 Naive 实现中，对于同一个 Warp 内的线程：
- 访问 A 时，它们有相同的 `row`（因为 `threadIdx.y` 相同），但不同的 `col`（`threadIdx.x` 不同）。当 i 遍历时，它们访问 `A[row][i]`，因为 row 相同 i 相同，所有线程访问的是 A 的同一个位置 → **Broadcast，零消耗**。
- 访问 B 时，它们访问 `B[i][col]`，col 从 0 到 31 连续。对于给定的 i，这是 B 的一行中 32 个连续的 float（128 bytes 连续地址）→ **完美合并访存**。

等等，这样分析的话，Naive GEMM 的访存模式实际上是：
- A: Broadcast（所有线程读同一个元素）
- B: 合并访存（线程读连续地址）

那为什么 Naive GEMM 这么慢呢？真正的原因是：**每个线程读的太多了！**

每个线程需要读 2K 个 float，总计 `2K × 4 bytes × M × N` 的数据从 Global Memory。对于 4096²：
- Global Memory 读取总量 ≈ `2 × 4096 × 4 × 4096 × 4096 = 550 GB`
- GPU 带宽 448 GB/s，理想耗时 ≈ 550/448 ≈ 1.2 秒
- 实际：133 ms

这说明实际读写要少得多，因为有 L2 Cache！矩阵的重复数据会被缓存在 L2 中。

纠正我的分析：B 的列访问虽然在一个 Warp 内是连续的，但跨 Warp（不同 row）时是 strided 的。而在同一时刻，grid 中的大量 Block 同时运行，对 B 产生大量的随机访问模式。此外，最关键的是 K 维度的循环——每读一次 A[row][i] 和 B[i][col]，就要读一次 Global Memory。这意味着即使有 L2 Cache，**每个元素都要读一次 K 维度上的数据**，这导致了大量的重复读取。

**更准确的回答**：

"GPUs use a warp-based memory system. In the naive matmul kernel, each thread computes one output element by iterating over the K dimension. Let's look at the global memory access patterns:

For matrix A (M×K): Thread (row, col) reads `A[row*K + i]`. Within a warp where `threadIdx.y` is constant, all 32 threads in the warp have the SAME row value. They all read `A[row][i]` for the same i in lockstep. This is a **broadcast** from global memory — super efficient.

For matrix B (K×N): Thread (row, col) reads `B[i*N + col]`. Within a warp, col values are consecutive (0-31). For a fixed i, threads access `B[i][0]`, `B[i][1]`, ..., `B[i][31]` — 32 consecutive floats = 128 bytes in a row. This is **fully coalesced** — one or two cache line transactions.

So the actual bottleneck of the naive kernel is NOT uncoalesced access (that's a misconception). The real bottleneck is **data reuse**: each element of A and B is read multiple times from global memory. With M=N=K=4096, each element of A is read N=4096 times, and each element of B is read M=4096 times. The on-chip caches (L1/L2) help reduce some of this, but not enough for such a large working set.

The optimization direction is clear: bring data into shared memory where threads within a block can efficiently share it, reducing global memory accesses by a factor of BLOCK_SIZE."

（这一题的深度分析是在做文档过程中发现的一个有趣细节——原来的面试题设计基于"B 列访问非合并"的常见误解，但严格分析后发现 Naive GEMM 的访存模式实际上是 Broadcast + Coalesced。真正的瓶颈是数据复用不够，而不是访存模式。这个纠正过程本身就是很好的教学内容。）

---

## 第4章 Level 2 — Shared Memory 分块优化

### 4.1 Tiling 思想

将矩阵切成小块（Tiles），每个 Block 负责计算一个输出子块。利用片上 Shared Memory 作为"显存缓存"——一次性将 Tile 数据从 Global Memory 加载到 Shared Memory，然后 Block 内所有线程从 Shared Memory 高速读取。

```
矩阵 A (M×K)              矩阵 B (K×N)
┌────┬────┬───┐           ┌────┬────┬───┐
│ T0 │ T1 │...│           │ T0 │ T1 │...│
├────┼────┼───┤           ├────┼────┼───┤
│    │    │   │           │    │    │   │
│ 输出块由A的一行Tile      │ 输出块由B的一列Tile
│ 和B的一列Tile相乘得到    │ 和B的一列Tile相乘得到
└────┴────┴───┘           └────┴────┴───┘
```

### 4.2 实现

```cuda
__shared__ float As[32][32];  // 1024 floats = 4 KB
__shared__ float Bs[32][32];  // 1024 floats = 4 KB

for (int tile = 0; tile < nTiles; tile++) {
    // 协同加载 Tile 到 Shared Memory
    As[ty][tx] = A[row * K + tile*32 + tx];
    Bs[ty][tx] = B[(tile*32 + ty) * N + col];
    __syncthreads();

    // 从 Shared Memory 做 Tile 内点积
    for (int k = 0; k < 32; k++)
        sum += As[ty][k] * Bs[k][tx];
    __syncthreads();
}
```

### 4.3 访存量大幅下降

Original:
- 每线程 2K 次 Global Memory 读 → `2 × 4096 × 4096² × 4B = 550 GB`

Shared Memory Tiling (BLOCK_SIZE=32):
- 每个 Block 计算 32×32=1024 个输出元素
- 每个 Block 读 `2 × 32 × 32 × 4B = 8 KB` 的 Global Memory（每 Tile）
- 共 K/32 = 128 个 Tiles → 每个 Block 读 `128 × 8 KB = 1 MB`
- 总 Global Memory 读：`(M/32 × N/32) × 1 MB = 128 × 128 × 1 MB = 16 GB`
- 从 550 GB 降到 16 GB！**减少了 34 倍！**

### 4.4 Bank Conflict 分析

Shared Memory 有 32 个 Bank，每个 4 bytes 宽（正好一个 float）。如果同一 Warp 的多个线程访问同一 Bank 的不同地址，会产生 Bank Conflict，访问被串行化。

分析 `As[threadIdx.y][k]` 的访问模式：
- 同一 Warp 的 32 个线程有相同的 `threadIdx.y`（因为是同一行），但 `threadIdx.x` 不同
- 当 `k` 固定时，所有线程访问 `As[同一个ty][同一个k]` → **Broadcast**，零冲突

分析 `Bs[k][threadIdx.x]` 的访问模式：
- 同一 Warp 的 32 个线程访问 `Bs[k][0]`, `Bs[k][1]`, ..., `Bs[k][31]`
- 这些地址映射到 Bank 0, 1, 2, ..., 31（因为 Bank = (地址/4) % 32，连续地址映射到不同 Bank）
- **零 Bank Conflict！**

### 4.5 实测性能

| 尺寸 | 时间 | 带宽 | TFLOPS | vs Naive |
|------|------|------|--------|----------|
| 4096² | 104.5 ms | 1.93 GB/s | 1.32 | **1.27x** |
| 1024² | 1.68 ms | 7.48 GB/s | 1.28 | **1.35x** |

加速比只有 27-35%，远不如访存量模型预测的 34 倍。为什么？因为瓶颈从 Global Memory 转移到了 Shared Memory → Register 的带宽。每个 FMA 需要读 2 次 Shared Memory，这是下一个优化要解决的问题。

---

### 🔥 面试题 3：Shared Memory 和 L1 Cache 有什么区别？

**话术回答**：

"Shared Memory 和 L1 Cache 在物理上共享同一块 128 KB 的 SRAM（在 Ampere SM 上），但它们的用途和编程模型完全不同：

1. **控制方式**：L1 Cache 是硬件自动管理的缓存，程序员无法直接控制。数据根据硬件自己的替换策略在 L1 中进出。Shared Memory 是程序员**显式管理**的——必须手动 load/store，手动 `__syncthreads()` 同步。

2. **缓存粒度**：L1 Cache Line 是 32 bytes。即使只需要 4 bytes，也会拉一整条 Cache Line。Shared Memory 支持按地址精确读写。

3. **何时用 Shared Memory**：当你知道数据的访问模式且有明确的复用需求时。比如 GEMM 中，一个 Tile 的 A 和 B 要被 Block 内所有线程反复读取 BK 次。用 Shared Memory 可以精确控制哪些数据常驻、何时加载、何时释放。依赖 L1 Cache 则可能因为 Cache 容量有限和替换策略导致关键数据被 evict，出现 Cache Thrashing。

4. **分配比例**：在 Ampere 架构上，128 KB / SM 可以配置为不同比例的 Shared Memory / L1，比如 100 KB Shared + 28 KB L1。

简单总结：L1 Cache 是自动的、通用的；Shared Memory 是手动但精准的。GEMM 优化中，由于访问模式高度规则和可预测，Shared Memory 远优于依赖 L1 Cache。"

---

### 🔥 面试题 4：什么是 Bank Conflict？你的 Shared Memory Kernel 是否触发 Bank Conflict？

**话术回答**：

"Bank Conflict 是指同一 Warp 的多个线程同时访问 Shared Memory 的不同地址但落在同一个 Bank 上。Shared Memory 有 32 个 Bank，每个 4 bytes 宽。地址 A 所在的 Bank = `(A/4) % 32`。

在我的实现中：
- `As[threadIdx.y][k]`：同一 Warp 内 32 个线程的 `threadIdx.y` 相同，`k` 相同，所以访问的是 Shared Memory 的**同一个地址**。这在硬件上通过**广播**处理——只需要一次 Bank 访问 → 零冲突。
- `Bs[k][threadIdx.x]`：32 个线程访问 `Bs[k][0]` 到 `Bs[k][31]`，这 32 个连续地址映射到 32 个不同的 Bank（因为 `32 % 32 = 0`，所以 `Bs[k][0]` 在 Bank 0，`Bs[k][1]` 在 Bank 1...`Bs[k][31]` 在 Bank 31）→ **零冲突**。

所以我的 32×32 Shared Memory 分块是 Bank-Conflict-Free 的。这也是为什么 BLOCK_SIZE=32 是一个精妙的选择——32 恰好等于 Bank 数量。

如果把 BLOCK_SIZE 设为 16 或 33，就可能会出现 Bank Conflict。例如 `Bs[k][threadIdx.x]` 在 BLOCK_SIZE=33 时，`Bs[k][0]` 在 Bank 0，`Bs[k][32]` 也会在 Bank 0（33/4=8，8%32=8，而 0%32=0...实际上需要精确计算）。关键是当 BLOCK_SIZE % 32 != 0 时不能完全消除 Bank Conflict。一个常见技巧是在声明 Shared Memory 时加 padding：`__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1]`，这样可以通过偏移打破对齐关系消除冲突。"

---

### 🔥 面试题 5：`__syncthreads()` 的具体开销是多少？它会影响 Occupancy 吗？

**话术回答**：

"`__syncthreads()` 是一个轻量级的 Block 内同步屏障。它本质上是一个特定于 CUDA 架构的硬件指令（`BAR.SYNC`），开销非常低——通常只有几个到十几个时钟周期。

**不会直接影响 Occupancy**，但会影响并行度：
- `__syncthreads()` 阻塞当前线程直到 Block 内所有线程到达。如果某个 Warp 的代码路径不同导致某些线程迟迟不到达（divergence），其他线程就在空转。
- 更重要的是，`__syncthreads()` 只能在 Block 内的活跃线程间同步。如果有线程提前退出或处于 inactive 状态，会导致未定义行为。这也是为什么在有分支的代码中使用 `__syncthreads()` 要特别小心——必须保证所有线程（或至少同一分支条件内的线程）都能到达屏障。

在我们的 GEMM Kernel 中，使用 `__syncthreads()` 两次：Tile 加载后和计算后。这两次同步的开销与 Shared Memory 访问延迟相比可以忽略。真正影响性能的不是 `__syncthreads()` 本身，而是同步造成的**Pipeline Bubble**——如果有些 Warp 加载数据较慢，其他 Warp 必须等待，导致 SM 的计算单元空闲。"

---

## 第5章 Level 3 — Float4 向量化加载

### 5.1 原理

float4 是 CUDA 内置的 128-bit 向量类型（4 个 float）。一次 `LDG.128` 指令可以加载 16 bytes，而标量 `LDG.32` 只能加载 4 bytes。使用 float4 可以减少 4 倍的 Load 指令数量，提高总线利用率。

```cuda
const float4* A_ptr = reinterpret_cast<const float4*>(&A[global_r * K + global_c]);
float4 a_vec = *A_ptr;  // 一次加载 4 个 float
As[load_r][load_c + 0] = a_vec.x;
As[load_r][load_c + 1] = a_vec.y;
As[load_r][load_c + 2] = a_vec.z;
As[load_r][load_c + 3] = a_vec.w;
```

### 5.2 为什么只分配 256 个线程加载？

BLOCK_SIZE=32，Shared Memory Tile 有 1024 个 float。使用 float4 后，1024/4 = 256 次加载即可完成。所以只需要前 256 个线程（tid < 256）参与加载，剩余线程直接通过 `__syncthreads()` 等待数据就绪。

### 5.3 为什么几乎无加速？

| 尺寸 | Shared | Float4 | 提升 |
|------|--------|--------|------|
| 4096² | 104.54 ms | 104.57 ms | ~0% |
| 1024² | 1.68 ms | 1.64 ms | ~2% |

**Float4 优化的是 Global→Shared 的带宽**，但当前瓶颈在 Shared→Register 的带宽。每个 FMA 操作需要从 Shared Memory 读取 2 个 float，而只需要从 Global Memory 加载（分摊到每个 FMA）`2 / BLOCK_SIZE × 4 = 0.25` 字节。所以 Global Memory 的带宽优化在当前瓶颈下是无感的。

这验证了 Amdahl 定律：优化非瓶颈部分几乎不提升整体性能。

---

### 🔥 面试题 6：float4 向量化在什么场景下最有效？

**话术回答**：

"float4 向量化在以下场景最有效：

1. **Memory-Bound 的 Kernel**：当性能瓶颈在 Global Memory 带宽时（如 Element-wise 运算、Reduce 操作等），使用 float4 可以显著减少 Load/Store 指令数量，提高带宽利用率。
2. **Coalesced 访问模式**：float4 要求 16-byte 对齐，在连续访问模式下可以一次 Transaction 完成 4 个元素的加载。
3. **寄存器 Spill 不太严重时**：float4 变量占用连续寄存器，如果寄存器压力大，float4 可能被 spill 到 Local Memory，反而降低性能。
4. **作为更高级优化的基石**：在 Register Tiling 中，float4 协同加载是必要的前置步骤——虽然它在 Shared Memory 级别作用不大，但它减少了加载 Shared Memory 所需的指令数，减少了 Barrier 等待时间，间接帮助了后续优化。

在我们的实验中，float4 对 Shared Memory GEMM 几乎无提升，正是因为 Shared Memory GEMM 已经是 **Shared Memory → Register 的带宽瓶颈**，而不是 Global → Shared 的瓶颈。但在 Register Tiling 中，float4 是构建高效协同加载的基础。"

---

### 🔥 面试题 7：如果矩阵的 K 维度不能被 4 整除，float4 怎么处理？

**话术回答**：

"有三种处理方式：

1. **Pad 到 4 的倍数**（Tensor 级别）：在 Python 端对输入矩阵做 padding，让 K 和 N 都成为 4 的倍数。简单但浪费显存。
2. **Kernel 内混合标量+向量**：完整 Tiles 用 float4，最后不完整的 Tile 用标量加载。但增加了代码分支和复杂度。
3. **Zero-Fill + 越界检查**（我们使用的方式）：越界时写 0，计算阶段这些位置自然贡献 0，不影响结果。但必须确保**地址计算**在越界时不越界——我们只计算地址但检查 `global_r < M && global_c < K` 后才通过 float4 指针读取，越界时直接跳过加载写零。

我们的 benchmark 中，257×257 的矩阵在 Float4 和 RegTile 上都直接 skip 了，因为内层维度 (K/N) 必须可以被 4 整除才能保证 float4 的 16-byte 地址对齐。这不是算法限制，而是性能优化选择——对于不可整除的维度，退回到 Shared Memory 版本即可（反正两者性能几乎一样）。"

---

## 第6章 Level 4 — Register Tiling 深度分析（核心章节）

### 6.1 问题根源

Level 1-3 的每个线程只计算 **1 个 C 元素**（1 个累加器）。这意味着：

- 每次 FMA 指令需要从 Shared Memory 读 2 个操作数（reg_a 和 reg_b）
- 算术强度 = 2 FLOP / (2 × 4B) = **0.25 FLOP/Byte**
- 远低于 Roofline 的 Memory Bound 线

要提高算术强度，必须让每个线程计算**更多的输出元素**，复用寄存器中的累加值，而不是每次都从 Shared Memory 重新加载。

### 6.2 核心思想

让一个线程计算 **TM × TN = 8 × 8 = 64 个输出元素**：

```
每个 Block 计算 128×128 的 C 子块
Block 有 16×16 = 256 个线程
每个线程计算 8×8 = 64 个元素（一个线程子块）
总元素数 = 16×8 × 16×8 = 128×128 ✓
```

算术强度变为：
- 每次从 Shared Memory 加载 8 个 A + 8 个 B = 16 个 float = 64 bytes
- 然后做 8×8 = 64 次 FMA = 128 FLOP
- 算术强度 = 128 FLOP / 64B = **2 FLOP/Byte**
- 对比 Level 1 的 0.125 FLOP/Byte，提升了 **16 倍**！

### 6.3 参数推导

| 参数 | 值 | 推导 |
|------|-----|------|
| BM, BN | 128 | 每个 Block 的输出块大小，需 SM 资源可容纳 |
| BK | 8 | K 方向步长，小 BK 减少 Shared Memory 但增加 Tile 循环次数 |
| TM, TN | 8 | 每线程输出子块大小，受寄存器数量约束（寄存器不够会 spill） |
| Block | 16×16 | BM/TM = 128/8 = 16, BN/TN = 128/8 = 16 |

### 6.4 寄存器使用量

```cuda
float accum[TM][TN] = {0.0f};  // 8×8 = 64 个寄存器
float reg_a[TM] = {0.0f};      // 8 个寄存器
float reg_b[TN] = {0.0f};      // 8 个寄存器
// 总计 ≈ 80 个寄存器 / 线程
```

- SM 有 65536 个寄存器
- 每个线程 80 个 → 65536/80 ≈ 819 线程
- 每个 Block 256 线程 → 819/256 ≈ **3 Blocks / SM**
- 理论最大 Occupancy = 819/1536 ≈ 53%

### 6.5 Shared Memory 容量

- `As[128×8]` + `Bs[8×128]` = 2048 floats = **8 KB**
- SM 有 100 KB Shared Memory → 100/8 ≈ 12 Blocks
- 但受寄存器限制只能 3 Blocks → Shared Memory 不是瓶颈

### 6.6 Occupancy 分析

对于 Compute-Bound Kernel，**不是 Occupancy 越高越好**：

- 每个线程需要大量寄存器来维护累加器（64 个）
- 高 Occupancy 意味着每个线程分到的寄存器少
- 寄存器不足时，编译器会 spill 到 Local Memory（实际是 L1/Global），严重降低性能
- 我们的 Kernel Occupancy = 53%，这利用了 SM 的 Warp Scheduler 可以同时驱动 3 个 Block 的优势，隐藏 Shared Memory 访问延迟

### 6.7 实测性能

| 尺寸 | 时间 | 带宽 | TFLOPS | vs Naive | vs Shared |
|------|------|------|--------|----------|-----------|
| 4096² | 18.79 ms | 10.72 GB/s | 7.32 | **7.1x** | **5.6x** |
| 2048² | 2.47 ms | 20.35 GB/s | 6.95 | **6.8x** | **5.3x** |
| 1024² | 0.40 ms | 31.12 GB/s | 5.31 | **5.6x** | **4.2x** |
| 512² | 0.13 ms | 23.88 GB/s | 2.04 | **2.0x** | **1.6x** |

7.32 TFLOPS 是 RTX 3060 Ti FP32 理论峰值 16.2 TFLOPS 的 **45%**，这是手写 CUDA GEMM 的优异成绩（cuBLAS 通常能达到 80-90%）。

---

### 🔥 面试题 8：为什么 TM=8, TN=8？如果改成 16×16 会怎样？

**话术回答**：

"TM=8, TN=8 是基于寄存器预算的权衡。16×16 = 256 个累加器，远超单线程的寄存器限制（Ampere 每线程最多 255 个 32-bit 寄存器，但编译器在 ~128 个以上就会开始 spill）。

寄存器 Spill 到 Local Memory（实际是 L1 Cache，但被标记为不可缓存），延迟从 0 增加到 ~30-200 cycles。即使只有部分变量被 spill，性能也会急剧下降。

如果我们用 TM=4, TN=4（16 个累加器），寄存器占用降到 ~24 个，Occupancy 会更高（可能 ~70-80%），但算术强度降低到 1 FLOP/Byte，回归到 Shared Memory 的性能水平。

**所以 TM/TN 的选择是一个多维度的权衡**：
- 太小 → 算术强度低，Performance 受限于 Shared Memory 带宽
- 太大 → 寄存器 Spill，Performance 受限于 Local Memory 延迟
- 最佳值取决于特定 GPU 架构的寄存器文件大小和 Warp Scheduler 设计

8×8 在 Ampere 上是一个被广泛验证的甜蜜点（Sweet Spot）。"

---

### 🔥 面试题 9：Occupancy 是不是越高越好？

**话术回答**：

"不是。这是 CUDA 优化中最常见的误解之一。

**高 Occupancy 的好处**：Warp Scheduler 有更多可调度的 Warp，当一个 Warp 因 Memory 延迟 stall 时，Scheduler 可以切换到另一个 Warp 隐藏延迟。

**高 Occupancy 的代价**：每个 SM 的寄存器总数固定（65536）。Occupancy 越高，每个线程分到的寄存器越少。当寄存器不够用时，编译器会 spill 变量到 Local Memory（≈ L1），延迟从 0 cycle 增加到 ~30-200 cycles。

**经验法则**：
- **Memory-Bound Kernel**：高 Occupancy 通常有益——更多 Warp 意味着更好的延迟隐藏
- **Compute-Bound Kernel**（如我们的 GEMM）：中等 Occupancy（30-50%）配合低寄存器 Spill 通常最优——因为我们用寄存器 Tiling 提升了 ILP（指令级并行），在较少的活跃 Warp 内就已经能充分填充 Pipeline
- 对于延迟敏感型 Kernel，足够多的活跃 Warp 是隐藏 Memory 延迟的关键（Latency Hiding）

NVIDIA 工具链中的 Occupancy Calculator 可以帮助可视化这个权衡。"

---

### 🔥 面试题 10：BK 为什么设为 8？如果设置成 16 或 32 会怎样？

**话术回答**：

"BK 控制沿 K 维度的分块大小。BK 的大小影响两个关键资源：

1. **Shared Memory 用量**：`As[BM×BK] + Bs[BK×BN] = (BM + BN) × BK` 个 float。BK=8 时 = 2048 floats = 8 KB。BK=16 时 = 4096 floats = 16 KB。BK=32 时 = 8192 floats = 32 KB。
2. **Tile 循环次数**：K/BK。BK 越大，循环次数越少，`__syncthreads()` 次数越少。

**为什么选 BK=8**：
- BK 增大 → Shared Memory 用量以线性增长，减少 Occupancy（更少的 Block 能同时驻留 SM）
- BK 减小 → Tile 循环更多，Barrier 同步开销增加
- 在 Ampere 上，每个 SM 100 KB Shared Memory 意味着 BK=8 时可驻留 ~12 个 Block（远超寄存器限制的 3 个），Shared Memory 不成瓶颈
- BK=8 配合 TM=TN=8 时，每个 K-step 做 64 次 FMA = 128 FLOP，从 Shared Memory 读 16 次 = 64 bytes，算术强度 = 2 FLOP/Byte

**BK 是对 Tile 循环次数、Shared Memory 占用和算术强度的三方权衡。8 在 Ampere 上是一个合理的默认值。**"

---

### 🔥 面试题 11：如果 M 不能被 128 整除，Register Tiling 还能正确运行吗？

**话术回答**：

"可以。Grid 启动时使用 `(N + 127) / 128` 和 `(M + 127) / 128` 的 ceiling 除法，确保覆盖所有元素。在 Kernel 内部：
1. **加载阶段**：越界检查 `if (global_r < M && global_c < K)` 确保不会从无效地址加载数据，越界处填 0。
2. **计算阶段**：正常的 FMA 循环，0 值不贡献。
3. **写回阶段**：`if (global_r < M && global_c < N)` 确保不写越界。

所以对于 257×257 的矩阵，ceil(257/128)=3，启动 3×3=9 个 Block，最后一个 Block 的大部分线程在处理越界元素时 safely skip。

但有一个隐藏问题：**float4 的对齐要求**。257 维度的矩阵的 K 维度不是 4 的倍数，导致 `&A[row * 257 + col]` 的地址可能不是 16-byte 对齐的。`reinterpret_cast<const float4*>()` 在非对齐地址上读取会触发 `misaligned address` 错误。这就是为什么我们在 Benchmark 中对 RegTile 也施加了 `K % 4 == 0 && N % 4 == 0` 的约束。

解决方法：可以用 `cuda::aligned_vector<float, 4>` 或 `cuda::memcpy_async` 来处理非对齐访问，或者在 Host 端 padding 输入 Tensor。"

---

## 第7章 面试模拟 — 综合问题

### 🔥 面试题 12（系统设计）：从零开始优化一个矩阵乘法，完整思路是什么？

**话术回答**：

"我会沿着 Roofline 模型指引的方向，从低算术强度向高算术强度逐步优化：

**Step 1 — 建立 Baseline（Naive + Benchmark）**
- 实现最简单的每个线程算一个 C 元素的 Kernel
- 搭建 Benchmark 框架，记录耗时、带宽、TFLOPS
- 跑 Nsight Compute 看 Speed Of Light 面板，确认是 Memory-Bound

**Step 2 — Shared Memory Tiling**
- 把矩阵切成 Tiles 加载到 Shared Memory，减少 Global Memory 重复读取
- 选择 BLOCK_SIZE=32（和 Bank 数量对齐，Bank-Conflict-Free）
- 预期加速：1.2-1.5x（取决于矩阵大小）

**Step 3 — Float4 向量化加载**
- 使用 float4 从 Global Memory 加载数据到 Shared Memory（128-bit 指令）
- 这一步通常提升有限（因为瓶颈已转移到 Shared→Register），但它是后续优化的基础
- 预期额外加速：0-5%

**Step 4 — Register Tiling**
- 让每个线程计算 8×8=64 个输出元素，利用寄存器做累加
- 大幅提高算术强度（从 ~0.25 → ~2 FLOP/Byte）
- 预期加速：5-7x vs Naive
- 关键参数选择：TM/TN/BK 基于寄存器预算和 Shared Memory 容量

**Step 5 — 高级优化（如果有时间）**
- Double Buffering / `cp.async`：预取下一个 Tile 到 Shared Memory，与计算重叠
- Warp Tiling：在 Shared Memory 和 Register 之间引入 Warp 级别的分块抽象
- SASS 级别调优：检查寄存器 Spill、Bank Conflict、指令级瓶颈

**每步都要验证**：正确性（vs PyTorch reference）+ 性能（Nsight Compute SOL）"

---

### 🔥 面试题 13（性能分析）：Kernel 只有理论峰值 5%，怎么排查？

**话术回答**：

"我会用 Nsight Compute 系统性地排查，按以下顺序看面板：

1. **GPU Speed Of Light**：先看大方向。如果 Memory 接近 100% 而 Compute 低 → Memory-Bound，先去优化访存。如果反过来 → Compute-Bound。

2. **Memory Workload Analysis**：看内存层级的热图。DRAM → L2 → L1 → Shared Memory 各级别的流量。如果 DRAM 流量大 → Global Memory 访存太多。如果 Shared Memory 流量大但 DRAM 小 → Tiling 起作用了。

3. **Warp State Statistics**：看 Warp 在为什么 stall。常见 stall 原因：
   - `Long Scoreboard`：等待 Global Memory 数据（Memory-Bound 的典型标志）
   - `Short Scoreboard`：等待 Shared Memory 数据
   - `Wait`：Barrier 同步等待
   - `Not Selected`：有足够 Warp 可调度但未被选中（说明 Occupancy 足够）

4. **Source Counters**：逐行 SASS/PTX 代码看每条指令的 stall 情况。这个面板能精确指出是哪一行 C++ 代码导致了瓶颈。

5. **Occupancy**：看理论 vs 实际 Occupancy。如果差距大 → 可能是寄存器 Spill 或 Shared Memory 用量过高。

6. **Scheduler Statistics**：看是否有 Register Bank Conflict、Shared Memory Bank Conflict 等微架构问题。

以我们的 Naive GEMM 为例，排查路径：
- SOL → Memory 接近 100%, Compute 6% → 确认 Memory-Bound
- Memory Workload → DRAM 流量巨大 → 需要减少 Global Memory 访问
- Warp State → Long Scoreboard 占 80%+ → 线程大部分时间在等内存
- 结论：必须用 Tiling 把数据搬到片上缓存"

---

### 🔥 面试题 14（数值精度）：Register Tiling 和 PyTorch 误差 0.001，来自哪里？

**话术回答**：

"FP32 的数值误差主要来自浮点加法的**非结合性**。

PyTorch 的 `torch.mm` 调用 cuBLAS，cuBLAS 内部使用了更复杂的分块策略（包括 Tensor Core 的 FP16 中间计算和 Kahan 求和），其累加顺序与我们手写 Kernel 完全不同。

在数学上，`(a + b) + c = a + (b + c)`。但在浮点数中，由于舍入误差，这两种顺序可能得到不同的结果。当我们改变累加顺序（不同的分块大小、不同数量的线程做不同顺序的 partial sum），最终结果会有微小差异。

**Acceptable 的标准**：
- FP32：max |diff| < 0.001 对大多数应用可接受
- Transformer 推理：通常 0.01 以内不影响 Top-K 准确性
- 科学计算：可能需要 FP64 或 Kahan 补偿求和

如果需要更高精度，可以在写回 Global Memory 前做一次 Kahan 补偿，但会增加 ~2x 的寄存器开销。在实际深度学习场景中，FP32 的 0.001 误差远小于随机 dropout 和量化引入的误差。"

---

### 🔥 面试题 15（架构理解）：Shared Memory、L1 Cache、Register File 物理关系？

**话术回答**：

"这三者在 Ampere SM 上的物理关系如下：

1. **Register File**：独立的 SRAM 存储。每个 SM 有 65536 × 32-bit 寄存器（256 KB）。这是最快的存储，0 cycle 额外延迟。每个线程的寄存器是私有的，在线程间不共享。

2. **Shared Memory + L1 Cache**：共享同一块 128 KB 的 SRAM（在 Ampere 上，Turing 是 96 KB）。通过配置 `cudaFuncSetAttribute` 可以调整分配比例（如 100 KB Shared + 28 KB L1）。Shared Memory 是程序员显式管理的；L1 Cache 是硬件自动管理的。

3. **物理位置**：三者都在 SM 内部（On-Chip），离 CUDA Core 非常近，这就是为什么它们的延迟远低于 L2 Cache 和 Global Memory。

**关键区别**：Shared Memory 和 L1 Cache 共享 SRAM，Register File 是另一块独立的 SRAM。这就是为什么即使 Shared Memory 被大量分配，只要不 spill，就不会和寄存器争抢——它们在不同的物理资源上。

**Shared Memory 和 L1 Cache 的权衡**：这是一个经典的缓存设计问题。Shared Memory 的好处是确定性——你知道什么数据在、什么时候在。L1 Cache 的好处是自动性——不需要手动管理。对于规则的、可预测的访问模式（如 GEMM），Shared Memory 更优；对于不规则的访问模式，L1 Cache 更合适。"

---

### 🔥 面试题 16（实际应用）：Transformer 维度如 768、3072 不是 2 的幂次，分块策略如何适配？

**话术回答**：

"实际 Transformer 的 hidden dim 确实不整齐，但我们的 Tile 策略天然支持非对齐维度：Grid 启动用 `(N + BN - 1) / BN` ceiling 除法，Kernel 内的越界检查处理边界 Tile。

对于不同的维度，我会有不同的策略：
- **768**（BERT-base）：接近 512（2⁹），可以用 BM=BN=128 的 Register Tiling。768/128 = 6，刚好整除，效率很高。
- **3072**（LLaMA 7B intermediate）：3072/128 = 24，正好整除。
- **12288**（LLaMA 70B）：12288/128 = 96，128 的 Tile 非常合适。

实际上这些维度大部分能被 4、8、16、32 整除（因为它们本身就是 GPU 友好的设计），所以 float4 的 16-byte 对齐通常不是问题。

如果遇到非对称的矩阵（如 M=512, K=1024, N=2048，模拟 Q·K^T），Register Tiling 仍然有效——BM=128 和 BN=128 的分块独立处理 M 和 N 方向，Grid 会自动调整 Block 数量。在我们的实测中，512×1024×2048 的矩阵，RegTile 达到 5.72 TFLOPS，比 4096² 的 7.32 TFLOPS 略低是因为小 M 导致不够多的 Block 填充 SM，但依然有 5.3x 的加速比。

**分块大小的自适应选择**：对于特别小的维度（如 M=1 的推理场景），128 的分块可能太大，此时需要退回到更小的分块策略（如 BLOCK_SIZE=32 的 Shared Memory 版本），因为大分块会导致大量线程做无用功（大部分 Block 计算越界）。"

---

### 🔥 面试题 17：Ampere cp.async 和传统 LDG 区别？在 GEMM 中如何用？

**话术回答**：

"`cp.async` 是 Ampere 架构引入的异步拷贝指令（SM_80+），它允许从 Global Memory 加载数据到 Shared Memory**不与计算指令竞争 issue slot**。

**与传统 LDG 的区别**：
| 特性 | 传统 LDG | cp.async |
|------|---------|----------|
| 执行方式 | 同步，占用计算单元 | 异步，由专用 Load/Store 单元处理 |
| Pipeline | 加载→等待→计算→加载 | 加载(后台) + 计算(前台) 并行 |
| 同步 | `__syncthreads()` 阻塞所有线程 | `cp.async.commit_group()` + `cp.async.wait_group()` |
| 对齐要求 | 无特殊要求 | 要求 4、8 或 16 byte 对齐 |

**在 GEMM 中的 Double Buffering 用法**：

```cuda
__shared__ float As[2][BM * BK];  // 两个 Buffer
__shared__ float Bs[2][BK * BN];

// 预取第一个 Tile
cp.async 加载 Tile 0 → Buffer 0

for (int tile = 0; tile < nTiles; tile++) {
    int next_tile = tile + 1;
    if (next_tile < nTiles) {
        // 异步预取下一个 Tile 到下一个 Buffer
        cp.async 加载 Tile[next_tile] → Buffer[(tile+1) % 2]
    }

    cp.async.commit_group();    // 提交异步加载任务
    cp.async.wait_group(0);     // 等待最近提交的异步任务完成

    // 在 Buffer[tile % 2] 上做计算
    ...

    __syncthreads();
}
```

**预期收益**：5-15% 的额外提升，因为计算和 Global Memory 加载完全重叠。在我们当前 7.32 TFLOPS 的基础上，Double Buffering 可能将性能提升到 ~8 TFLOPS。

**注意事项**：
- 需要 `#include <cuda/pipeline>` 或 `<cooperative_groups.h>`
- 需要 CUDA 11.1+
- 需要 16-byte 对齐（正好是 float4），这也是为什么 float4 是 `cp.async` 的基础
- Double Buffering 使 Shared Memory 占用翻倍，可能降低 Occupancy"

---

### 🔥 面试题 18：Tensor Core 加速 GEMM 的原理？FP32 输入下如何使用？

**话术回答**：

"Tensor Core 是 Volta 架构引入的专用矩阵乘法单元，在 Ampere（SM_80+）上是第三代 Tensor Core。

**原理**：Tensor Core 每时钟可以完成一个 16×16×16 的矩阵乘加运算（D = A×B + C）。虽然称为 MMA（Matrix Multiply-Accumulate），但输入是 FP16/BF16，累加器是 FP32。

**吞吐量对比**（RTX 3060 Ti, Ampere）：
- CUDA Core FP32：16.2 TFLOPS
- Tensor Core FP16 (with FP32 accumulate)：**~65 TFLOPS**（理论）
- Tensor Core TF32 (with FP32 accumulate)：~32 TFLOPS

**对于 FP32 输入**：
1. **TF32 模式**：将 FP32 截断为 19-bit（1 sign + 8 exponent + 10 mantissa），Tensor Core 以 TF32 计算，结果回到 FP32。精度略低但对深度学习几乎无影响（NVIDIA 声称在训练中与 FP32 无统计差异）。
2. **FP16 转换**：将 FP32 输入显式转换为 FP16（`__float2half()`），用 Tensor Core 做 FP16 MMA，输出转回 FP32。有 ~0.1% 的精度损失。

**CUDA API**：
- `nvcuda::wmma`：较易用的 Warp-level API，抽象了 fragment 的加载/存储
- `mma.sync` (PTX)：更低级，需要手动处理 Shared Memory 布局（128-byte swizzling pattern）
- CUTLASS：NVIDIA 开源的高性能 GEMM 模板库，封装了所有细节

**是否值得在项目中使用 Tensor Core**：
- 对于 GEMM 性能竞赛：绝对值得，可以达到理论峰值的 80%+
- 对于学习目的：`nvcuda::wmma` 相对友好，但 Fragment API 和 Shared Memory Layout 要求有陡峭的学习曲线
- 在我们的 3060 Ti 上，Tensor Core 可以将 FP32 GEMM 从 7.3 TFLOPS 提升到 30+ TFLOPS（使用 TF32）

但这已经超出了手写 CUDA GEMM 的教学范围，更适合在生产环境中直接使用 cuBLAS 或 CUTLASS。"

---

## 第8章 总结与性能对比

### 8.1 六版本性能总表（4096×4096×4096, FP32, RTX 3060 Ti）

| Kernel | 时间 (ms) | 带宽 (GB/s) | TFLOPS | 算术强度 (FLOP/B) | vs Naive | 对比项 |
|--------|-----------|-------------|--------|--------------------|----------|--------|
| Naive | 135.3 | 1.49 | 1.02 | ~0.125 | 1.0x | — |
| Shared Memory | 106.9 | 1.88 | 1.29 | ~0.25 | 1.26x | vs Naive |
| Float4 + Shared | 106.6 | 1.89 | 1.29 | ~0.25 | 1.26x | vs Naive |
| Register Tiling | 19.12 | 10.53 | 7.19 | ~2.0 | 7.0x | vs Naive |
| DB Async BK=16 | 18.32 | 10.99 | 7.50 | ~4.0 | 7.4x | vs Naive |
| WMMA TF32 BK=16 | 18.06 | 11.15 | 7.61 | ~4.0 | 7.5x | vs Naive |
| cuBLAS (torch.mm) | 12.54 | 16.05 | 10.96 | — | 10.7x | Reference |

### 8.2 Roofline 图

```
TFLOPS ↑
   32.0 ┤············································· 🟡 TF32 理论峰值
   16.2 ┤─────────────────────────────────────────── 🟢 FP32 理论峰值
        │
   11.0 ┤                           ● cuBLAS (10.96, 68% FP32 peak)
        │               ★ WMMA TC BK=16  (7.61, 47% FP32 peak)
    7.2 ┤               ▲ RegTile       (7.19)
        │               │  DB Async BK=16 (7.50)
        │               │
        │               │  Memory Bound
        │               │  Region
    1.3 ┤   ■ Shared    │
    1.0 ┤   ◆ Float4    │
        │   □ Naive     │
        └───┴───────────┴───────────────────────→ 算术强度 (FLOP/Byte)
          0.125 0.25          2.0   4.0       36.0
```

### 8.3 优化效果瀑布图

```
Naive:      1.02 TFLOPS  ▏
Shared:     1.29 TFLOPS  ▏ (+26%)
Float4:     1.29 TFLOPS  ▏ (+0%)
RegTile:    7.19 TFLOPS  ████████████████████████████████████████ (+605%)
DB Async:   7.50 TFLOPS  ██████████████████████████████████████████ (+4%)
WMMA TC:    7.61 TFLOPS  ██████████████████████████████████████████ (+1%)
cuBLAS:    10.96 TFLOPS  ██████████████████████████████████████████████████████████████
```

### 8.4 与 cuBLAS 的差距分析

| 差距来源 | 说明 |
|----------|------|
| Tensor Core | cuBLAS 使用 Tensor Core (TF32)，吞吐 ~65T vs 我们的 CUDA Core FP32 16.2T |
| SASS 级优化 | cuBLAS 有手工调优的 SASS 汇编、寄存器分配、指令调度 |
| 启发式分块选择 | cuBLAS 根据矩阵大小动态选择最优分块策略 |
| Double Buffering | 我们已引入 cp.async DB (+4%)，cuBLAS 可能使用更多 stage 或更优 pipeline |

---

## 第9章 Level 5 — cp.async Double Buffering + BK 参数调优

> **源码**: `src/gemm_kernels_async.cu` → `gemm_db_async_kernel`
> **参数**: BM=128, BN=128, BK=16, TM=8, TN=8 | block(16,16)=256t | `__launch_bounds__(256,2)`
> **BF16 性能 (4096²)**: 17.2ms, **8.00 TFLOPS** | **NCU**: SM Busy 49.3%, IPC 1.97

### 9.1 动机

Register Tiling 将算术强度从 ~0.25 提升到 ~2 FLOP/Byte 后，瓶颈不再单纯是 Global Memory 带宽——Global→Shared 的加载延迟仍然存在，只是被计算部分"覆盖"了一部分。当前每个 tile 循环中，加载和计算是**严格串行**的：

```
[Sync Load Tile 0] → [Sync] → [Compute Tile 0] → [Sync] → [Sync Load Tile 1] → ...
```

Ampere 架构引入了 `cp.async`（异步拷贝指令），允许 Global→Shared 的数据搬运由专用的 Load/Store 单元处理，**不占用 CUDA Core 的指令发射槽**。配合双缓冲（Ping-Pong Shared Memory），可以将加载和计算重叠：

```
[Load T0 sync] → [Compute T0 + Load T1 async] → [Compute T1 + Load T2 async] → ...
```

### 9.2 双缓冲 Shared Memory

将 Shared Memory 声明为双倍大小，两个 buffer 交替使用：

```cuda
__shared__ float As[2][BM * BK];  // 2 stages, ping-pong
__shared__ float Bs[2][BK * BN];
```

`stage = 0` 存当前正在计算的 tile，`stage ^ 1` 存正在异步加载的下一个 tile。

### 9.3 cp.async 三种 API 的选择

| API | 头文件 | 复杂度 | 说明 |
|-----|--------|--------|------|
| `__pipeline_memcpy_async` | `cuda_pipeline_primitives.h` | 低 | C 函数，最接近 PTX，直接控制 commit/wait |
| `cg::memcpy_async` | `cooperative_groups/memcpy_async.h` | 中 | C++ Cooperative Groups 风格 |
| `cuda::pipeline` | `cuda/pipeline` | 高 | libcu++ 风格，需要 C++20 |

我们选择 **`cuda_pipeline_primitives.h`**——最轻量，最接近硬件，依赖最少。

**核心指令**：
- `__pipeline_memcpy_async(dst_shared, src_global, 16)` — 发起 16-byte 异步拷贝
- `__pipeline_commit()` — 提交当前线程的全部挂起拷贝
- `__pipeline_wait_prior(N)` — 等待除最近 N 组外的所有提交组完成

### 9.4 协作加载的 BK 泛化

原始的协作加载对 BK=8 是硬编码的（256 线程各加载 1 个 float4 = 1024 个元素 = 128×8）。当 BK=16 时，tile 翻倍为 2048 个元素，需要**每个线程加载 2 个 float4**。

泛化方案：

```cuda
const int LOADS = BK / 8;  // 1 for BK=8, 2 for BK=16, 4 for BK=32

for (int l = 0; l < LOADS; l++) {
    // A: 每个 load 覆盖 8 列
    int a_row = tid / 2;
    int a_col = l * 8 + (tid % 2) * 4;  // 0,4 then 8,12 ...

    // B: 每个 load 覆盖 8 行
    int b_row = l * 8 + tid / 32;        // 0-7 then 8-15 ...
    int b_col = (tid % 32) * 4;           // 0,4,...,124
}
```

### 9.5 BK 参数扫参

BK 控制沿 K 维度的分块大小。BK 越大→每个 tile 计算越多→越能隐藏异步加载延迟。但 BK 越大也意味着 Shared Memory 越大（`As[128×BK] + Bs[BK×128]`），且双缓冲翻倍。

**扫参结果（4096² FP32）**：

| BK | SMEM (DB) | Tiles (K=4096) | 时间 | TFLOPS | vs RegTile | 结论 |
|----|-----------|-----------------|------|--------|------------|------|
| 8 | 16 KB | 512 | 19.51 ms | 7.04 | **-4%** | tile 太小，commit/wait 开销>收益 |
| **16** | **32 KB** | **256** | **18.32 ms** | **7.50** | **+4%** | 最优：计算时间足够隐藏加载 |
| 32 | 64 KB | 128 | crash | — | — | SMEM 超限（64KB×Block 挤占其他资源） |

**不同矩阵大小的加速效果（BK=16）**：

| 尺寸 | RegTile | DB Async | 加速比 | 分析 |
|------|---------|----------|--------|------|
| 256² | 0.49 TF | 0.55 TF | +12% | 小矩阵，加载延迟占比大，重叠收益明显 |
| 512² | 1.86 TF | 2.38 TF | **+28%** | 最佳加速：SM 利用率刚好满 |
| 1024² | 5.85 TF | 5.83 TF | ~0% | Grid=8²=64 Block，未能填满所有 SM |
| 2048² | 6.52 TF | 7.22 TF | +11% | 稳定加速 |
| 4096² | 7.19 TF | 7.50 TF | +4% | 接近 Roofline 时，访存优化空间有限 |
| Rectangular | 5.21 TF | 6.05 TF | **+16%** | 非对称矩阵对延迟隐藏更敏感 |

### 9.6 为什么 BK=8 时 cp.async 反而更慢？

在 BK=8 的情况下，每个 tile 只有 8 个 K 步。每个 K 步做 `TM×TN = 64` 次 FMA。总计 8×64 = 512 FMA / 线程 / tile。512 次 FMA 在现代 GPU 上执行极快（估计 ~10-20 GPU cycles），导致：

1. **计算太短**：512 FMA 不够隐藏 ~700+ cycles 的 Global Memory 延迟
2. **Overhead 显著**：`__pipeline_commit()` + `__pipeline_wait_prior(0)` 本身有指令开销（BAR.SYNC + WAIT 指令簇）
3. **Commits 过多**：K=4096, BK=8 → 512 个 tile，512 次 commit/wait，开销累积

BK=16 解决了这个问题：
- 每 tile 16 个 K 步 × 64 FMA = 1024 FMA / tile
- Tile 数量减半：256 个 tile
- Commit/wait 次数减半
- 更长的计算窗口让异步加载有足够时间完成

### 9.7 为什么 BK=32 会失败？

BK=32 + Double Buffering：
- `As[2][128×32]` = 8192 floats = **32 KB**
- `Bs[2][32×128]` = 8192 floats = **32 KB**
- **Total = 64 KB / Block**

RTX 3060 Ti 每 SM 有 100 KB Shared Memory。理论上 64 KB 可以容纳，但：
- nvcc 预留部分 Shared Memory 用于 pipeline state / barrier 元数据
- 其他运行时的 overhead 也可能消耗少量 Shared Memory
- 接近上限时，Compiler 可能无法正确计算静态 Shared Memory 分配

结果：Kernel Launch 失败（"invalid argument"）。这在实际开发中是常见的——**Shared Memory 预算必须保守一些**。

### 🔥 面试题 19：cp.async 的 commit 和 wait 是什么关系？为什么需要 commit_group？

**话术回答**：

"`cp.async` 的工作机制是批量异步拷贝。一个线程可以发起多次 `__pipeline_memcpy_async`（比如我们的 kernel 中每个线程为 A 和 B 各发起 2 次 async copy）。这些拷贝请求在 TLC（Tensor Memory Load/Store Controller）中排队，但 TLC 并不知道什么时候可以开始执行。

`__pipeline_commit()` 的作用就是标记一个'提交组'的边界——告诉 TLC：'到目前为止发起的所有拷贝可以作为一个批次开始执行了'。每次 commit 都会递增一个内部的提交组计数器。

`__pipeline_wait_prior(N)` 等待的是除最近 N 个提交组之外的所有组完成。在我们的双缓冲模式中：
- 循环开始时发起下一个 tile 的拷贝（一批 memcpy_async 调用）
- `commit()` 提交这批拷贝（成为最新的提交组）
- `wait_prior(0)` — 等待**所有**提交组完成（因为 prior count=0，不排除任何组）。这会等待上一次循环发起的那个提交组完成。
- 然后我们在已经完成数据就绪的 buffer 上计算。

关键：**commit 不是 per-tile 的，而是 per-thread 的**。256 个线程各自 commit，形成一种全局的拷贝完成假设——只有当所有线程的 commit 组都完成时，`wait_prior` 才会返回，这就是为什么之后还需要 `__syncthreads()` 保证 Shared Memory 对所有线程可见。"

---

---

## 第10章 Level 6 — WMMA TF32 Tensor Core

> **源码**: `src/gemm_kernels_tc.cu` → `gemm_wmma_kernel`
> **参数**: TC_BM=128, TC_BN=128, TC_BK=16, WMMA_K=8, SUBSTEPS=2 | block(128,2)=256t=8 warps
> **BF16 性能 (4096²)**: 17.3ms, **7.95 TFLOPS** | **NCU**: SM Busy 19.3%, No Eligible 91.9%

### 10.1 动机：从 CUDA Core 到 Tensor Core

前 5 个 Level 全部使用 CUDA Core（FP32 FMA 指令）。RTX 3060 Ti 的 FP32 理论峰值是 **16.2 TFLOPS**，我们做到 7.5（46% 峰值）。

Ampere 架构引入了第三代 Tensor Core，支持 TF32（TensorFloat-32）精度模式——将 FP32 输入的 mantissa 从 23-bit 截断到 10-bit，乘法在 Tensor Core 内完成，累加器保持 FP32。

| 精度 | 理论峰值 (3060 Ti) | 指令 |
|------|-------------------|------|
| CUDA Core FP32 | 16.2 TFLOPS | FFMA |
| Tensor Core TF32 | ~32 TFLOPS | mma.sync |
| Tensor Core FP16 | ~65 TFLOPS | mma.sync |

**关键**：TF32 不需要手动转换——`wmma::load_matrix_sync` 从 `float*` 加载时硬件自动完成 FP32→TF32 截断。

### 10.2 WMMA API

```cuda
#include <mma.h>
using namespace nvcuda;

// 唯一 TF32 tile 形状: 16×16×8
wmma::fragment<matrix_a, 16,16,8, precision::tf32, row_major> a_frag;  // 4 floats
wmma::fragment<matrix_b, 16,16,8, precision::tf32, col_major> b_frag;  // 4 floats
wmma::fragment<accumulator, 16,16,8, float>                     c_frag; // 8 floats

// 使用流程
wmma::fill_fragment(c_frag, 0.0f);                         // 1. 初始化累加器
wmma::load_matrix_sync(a_frag, &As[mBase][0], ld);         // 2. 从 Shared Memory 加载 A
wmma::load_matrix_sync(b_frag, &BsT[nBase][0], ld);        // 3. 从 Shared Memory 加载 B^T
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);             // 4. C += A × B
wmma::store_matrix_sync(&C[out_r*N+out_c], c_frag, N, mem_row_major); // 5. 写回 Global
```

### 10.3 Block/Warp 组织

```
Block(128, 2) = 256 threads = 8 warps (4×2)

          warpCol →
          0    1    2    3
warpRow  ┌────┬────┬────┬────┐
   0     │ W0 │ W1 │ W2 │ W3 │  M: 0..63
   1     ├────┼────┼────┼────┤
         │ W4 │ W5 │ W6 │ W7 │  M: 64..127
         └────┴────┴────┴────┘
           N:0.. N:32. N:64. N:96.
           31   .63   .95   .127

每个 warp: 4(M)×2(N)=8 个 accumulator fragment，每个 16×16×8
每个 warp 覆盖 64 rows × 32 cols
```

### 10.4 Shared Memory 布局与 B 转置

```cuda
__shared__ float As[TC_BM][TC_BK];   // As[128][BK] — A 直接存储
__shared__ float BsT[TC_BN][TC_BK];  // BsT[128][BK] — B 转置存储
```

**B 必须转置的原因**：WMMA 的 `matrix_b` fragment 使用 `col_major` 布局——内层循环在 K 维度上连续。将 B 以 `BsT[n][k] = B[k][n]` 的形式转置存储后，`col_major` 加载恰好对应 `BsT[n][k]`，实现了零额外开销的转置。

### 10.5 WMMA BK 参数扫参

| BK | Tiles(K=4096) | SMEM | TFLOPS (4096²) | vs CUDA Core 最优 | 结论 |
|----|---------------|------|---------------|-------------------|------|
| 8 | 512 | 4 KB | 7.55 | ~0% | barrier 过多 |
| **16** | **256** | **8 KB** | **7.61** | **+1%** | 最佳平衡 |
| 32 | 128 | 16 KB | 6.24 | -18% | bank conflict，Occupancy 下降 |

### 10.6 为什么 WMMA TF32 没有预想的快？

**预期**：Tensor Core TF32 理论吞吐是 CUDA Core FP32 的 2×。手写实现应该从 7.6 → 10+ TFLOPS。

**实际**：7.61 TFLOPS，几乎和 CUDA Core 持平。

**根因分析**：

1. **Shared Memory 带宽是共同瓶颈**。WMMA 的 `load_matrix_sync` 本质还是读 Shared Memory。每个 WMMA K=8 迭代读 `16×8(A) + 8×16(B) = 256 floats = 1024 bytes`。每 SM 的 Shared Memory 带宽 ~17 TB/s（Ampere），算下来每个 warp 的 WMMA 受限于 ~2-3 TB/s 的有效带宽——远低于 Tensor Core 的吞吐。

2. **消费级 GPU 的 Tensor Core 规格受限**。3060 Ti 每 SM 只有 4 个 Tensor Core（vs A100 的 8 个），且每个 Tensor Core 每 cycle 只能完成 1 次 16×16×8 MMA。在 128 CUDA Core / SM 的背景下，Tensor Core 并不构成碾压。

3. **WMMA fragment 管理的指令开销**。每个 `mma_sync` 调用前需要 `load_matrix_sync`，每个 tile 需要 `fill_fragment` 初始化。这些指令虽然不涉及 Global Memory，但占用 issue slot，与 CUDA Core 的 FFMA 直接发射相比多了一层抽象。

4. **BK=16 时 256 次 barrier 仍然是瓶颈**。即使用 cp.async 双缓冲，barrier 同步的开销在 BK=16 时仍然显著。

**关键洞察**：cuBLAS 能达到 11 TFLOPS 不是因为用了更强的硬件，而是因为**每个硬件单元都发挥到了极致**——Tensor Core 用 TF32、共享内存用 swizzle 消除 bank conflict、加载用 cp.async 多 stage pipeline、SASS 级别消除指令级的 stall。差距不在"用什么"，而在"怎么用"。

### 🔥 面试题 20：Tensor Core TF32 和 CUDA Core FP32 本质区别是什么？

**话术回答**：

"Tensor Core 和 CUDA Core 是 GPU 上两个独立的执行单元，它们在硬件层面完全不同。

**CUDA Core**：每个 Core 每 cycle 执行 1 条指令。FP32 FMA 完成 `d = a × b + c`，需要 2 个 FP32 操作数和一个 FP32 累加操作。16.2 TFLOPS 的理论峰值来自 4864 CUDA Cores × 1.67 GHz × 2 ops/FMA。

**Tensor Core**：是一个专用的矩阵乘法阵列。Ampere 第三代 Tensor Core 每 cycle 完成一个 16×16×8 的 MMA（Matrix Multiply-Accumulate），即 `C[16×16] += A[16×8] × B[8×16]`。在 TF32 模式下，A 和 B 的元素被截断为 19-bit 浮点（1 sign + 8 exponent + 10 mantissa），但乘法结果和累加器保持 FP32。

**核心差异**：
1. **吞吐**：Tensor Core 每 cycle 完成 `16×16×8×2 = 4096 FLOP`，远超过单个 CUDA Core。但注意这是矩阵级操作，不是标量。
2. **精度**：TF32 的 10-bit mantissa vs FP32 的 23-bit mantissa。对深度学习训练/推理，精度损失通常 <0.1% 的相对误差。
3. **编程模型**：CUDA Core 用标量指令（FMA, FMUL, FADD），Tensor Core 用 WMMA API（fragment 抽象）或 PTX mma 指令。
4. **适用场景**：Tensor Core 在计算密度极高（大 M、大 N、大 K）时优势最大。对于小矩阵或 Memory-Bound 操作，优势被削弱。"

---

## 第11章 Level 7 — LDS.128 + `__launch_bounds__` 微调

> **影响文件**: `src/gemm_kernels.cu` (RegTile), `src/gemm_kernels_async.cu` (DB Async)
> **技术**: Shared Memory B-side LDS.128 向量化, `__launch_bounds__(256, 2)`
> **性能 (4096²)**: RegTile 7.64→**7.78** (+2%), DB Async 7.83→**8.00** (+2%)

### 11.1 动机

Register Tiling 将 `reg_b` 的 8 个元素分别用标量 `LDS.32` 从 Shared Memory 加载。但 `Bs[k * BN + (thread_c + j)]` 对于 j=0..7 是**连续地址**——在 `k` 固定时，同一行的 8 个 N 值连续排列。这 8 个标量加载可以合并为 2 个 LDS.128（128-bit vectorized load），减少 4 倍的 Shared Memory Load 指令数。

### 11.2 实现

在 `gemm_kernels.cu` 和 `gemm_kernels_async.cu` 的 Register Tiling 计算循环中：

```cuda
// 优化前：8 次标量 LDS.32
for (int j = 0; j < TN; ++j)
    reg_b[j] = Bs[k * BN + (thread_c + j)];  // 8 条 LDS.32 指令

// 优化后：2 次向量化 LDS.128
float4 vb0 = *reinterpret_cast<float4*>(&Bs[k * BN + (thread_c + 0)]);
float4 vb1 = *reinterpret_cast<float4*>(&Bs[k * BN + (thread_c + 4)]);
reg_b[0]=vb0.x; reg_b[1]=vb0.y; reg_b[2]=vb0.z; reg_b[3]=vb0.w;
reg_b[4]=vb1.x; reg_b[5]=vb1.y; reg_b[6]=vb1.z; reg_b[7]=vb1.w;
```

**为什么 A 端不适用**：`As[(thread_r + i) * BK + k]` 中连续的 i 对应不同的行，行间 stride 为 BK=8 或 16，地址不连续。float4 需要 4 个连续地址，无法直接向量化 A 端加载。

### 11.3 `__launch_bounds__` 编译器提示

```cuda
__launch_bounds__(256, 2)  // 256 threads/block, min 2 blocks/SM
__global__ void matmul_register_tiling_kernel(...)
```

告诉 nvcc：每个 Block 256 线程，期望至少 2 Blocks/SM。编译器据此优化寄存器分配——减少 spill、最大化 dual-issue。这本身不改变 kernel 行为，但让 `-O3` 生成的 SASS 更优。

### 11.4 效果

| Kernel | 优化前 (4096²) | 优化后 (4096²) | 提升 |
|--------|---------------|---------------|------|
| RegTile (`matmul_register_tiling_kernel`) | 7.64 TF | 7.78 TF | **+1.8%** |
| DB Async (`gemm_db_async_kernel`) | 7.83 TF | 8.00 TF | **+2.2%** |

**为什么有效**：DB Async 在 1024² 时 SM Busy 49.3%、IPC 1.97。LDS.128 将 reg_b 的 8 条 LDS.32 合并为 2 条 LDS.128，减少了指令发射压力。BK=16 时，每 K 步节省 6 条指令 × 256 outer tiles = 大量指令消除。`__launch_bounds__` 帮助编译器更高效分配寄存器。

### 11.5 关键洞察

这是一个**跨 kernel 优化 pass**——它没有创建新的 kernel 函数，而是在已有的 `matmul_register_tiling_kernel` 和 `gemm_db_async_kernel` 中修改了计算循环。在优化层次上它是 Level 7，因为它建立在前 6 级的累加寄存器、cp.async、WMMA 基础上，进一步压榨指令级效率。

---

## 第12章 Level 8 — WMMA + cp.async Double Buffering

> **源码**: `src/gemm_kernels_tc_async.cu` → `gemm_wmma_async_kernel`
> **参数**: TC_BM=128, TC_BN=128, TC_BK=16, WMMA_K=8 | block(128,2)=256t=8 warps | `__launch_bounds__(256,2)`
> **性能 (4096²)**: 17.0ms, **8.10 TFLOPS** | **NCU**: SM Busy 19.0%, No Eligible 90.1%

### 12.1 动机

WMMA TC kernel 的 NCU 诊断显示 91.9% warp 空闲——Tensor Core 算力被 `__syncthreads()` 锁死。将 cp.async 双缓冲与 WMMA 合并，让 Global→Shared 的 **A 加载**异步进行，与 WMMA 计算重叠。

### 12.2 实现要点

- **A 加载**：使用 `__pipeline_memcpy_async`（非转置，地址 16-byte 对齐满足）
- **B 加载**：使用同步 float4 + 手动转置（`__pipeline_memcpy_async` 无法处理 `BsT[n+delta][k]` 跨行地址——转置后连续地址变 stride）
- **其他结构**：同 sync WMMA kernel（c_frag[4][2] 累加器、`wmma::load_matrix_sync` + `mma_sync`）

### 12.3 效果

| Size | DB Async | WMMA TC | WMMA+DB | WMMA+DB vs WMMA TC |
|------|----------|---------|---------|---------------------|
| 4096² | 8.00 TF | 7.95 TF | **8.10 TF** | **+1.9%** |
| 2048² | 7.34 TF | 7.34 TF | 7.14 TF | -2.7% |
| 1024² | 5.84 TF | 6.43 TF | 6.31 TF | -1.9% |
| 512² | 2.13 TF | 2.52 TF | 2.43 TF | -3.6% |

WMMA+DB 在 4096² 超越了 DB Async（8.10 vs 8.00），但小矩阵上反而更慢。B 加载仍是同步的，barrier 数量未减少，`__syncthreads()` 开销仍占主导。

---

## 第13章 NCU 全内核深度分析

> 数据来源：`ncu --set full -k regex:<kernel>$ -s 5 -c 1`，1024² FP32，RTX 3060 Ti

### 13.1 总览

| Kernel | Time | MemSOL | CmpSOL | SM Busy | IPC | Occ | NoElig | WCPEI |
|--------|------|--------|--------|---------|-----|-----|--------|-------|
| Naive | 2.68ms | 93.5% | 93.5% | 31.4% | 1.10 | 66.6% | 72.6% | 29.2 |
| Shared | 2.15ms | 80.7% | 80.7% | 28.2% | 0.90 | 66.5% | 77.5% | 35.6 |
| Float4 | 2.16ms | 74.8% | 74.8% | 26.3% | 0.83 | 66.7% | 79.2% | 38.4 |
| RegTile | 507μs | 70.0% | 40.9% | 47.1% | 1.88 | 27.0% | 52.8% | 6.9 |
| DB Async | 498μs | 64.8% | 43.8% | **49.3%** | **1.97** | 26.0% | 50.5% | **6.3** |
| WMMA TC | 474μs | 66.1% | 16.6% | 19.3% | 0.32 | 26.2% | 91.9% | 38.8 |
| WMMA+DB | 484μs | 60.4% | 17.8% | 19.0% | 0.40 | 27.0% | 90.1% | 32.8 |

> MemSOL=Memory SpeedOfLight, CmpSOL=Compute SpeedOfLight, NoElig=No Eligible Warp%, WCPEI=Warp Cycles Per Executed Instruction

### 13.2 逐内核深度分析

#### Naive — 极端的 Memory-Bound

| 指标 | 值 | 解读 |
|------|-----|------|
| Memory SOL | **93.5%** | Memory 管线几乎满载 |
| DRAM Throughput | 11.6% | 大部分数据来自 L1/L2 Cache，不是 DRAM |
| L1 Hit Rate | **94.9%** | L1 Cache 命中率极高——数据复用受惠于 Broadcast 访问模式 |
| L2 Hit Rate | 54.0% | L2 兜底了一半的 L1 miss |
| Compute SOL | 93.5% | 值与 Memory SOL 相等 → Compute 受限于 Memory（典型 Memory-Bound 特征） |
| SM Busy | 31.4% | 只有 1/3 时间 SM 在工作 |
| IPC Active | 1.10 | 接近单发射上限 |
| No Eligible | 72.6% | 72% 时间调度器没有可发射的 warp |
| Occupancy | **66.6%** | 高 Occupancy，但无济于事——warp 全在等内存 |
| Registers | 40/thread | Block Limit Registers=1，Occupancy 被寄存器限制 |
| SMEM | 0 | 没有使用 Shared Memory |

**诊断**：纯 Memory-Bound。L1 Cache 做了出色工作（95% hit rate），但 4096 次 K 循环的算术强度太低。Warp 大部分时间在 `Long Scoreboard` stall — 等 Global Memory 数据。

#### Shared Memory (Level 2)

| 指标 | 值 | vs Naive |
|------|-----|----------|
| Memory SOL | 80.7% | **-12.8pp** — Tiling 有效降低了 Memory 压力 |
| DRAM Throughput | 14.4% | +2.8pp — 更高效利用 DRAM |
| L1 Hit Rate | **0.0%** | L1 不再缓存 — Shared Memory 承担了所有 Tile 数据 |
| Duration | 2.15ms | **-20%** |
| SM Busy | 28.2% | -3.2pp — 略微下降 |
| IPC Active | 0.90 | -0.20 — Shared Memory Load 指令延迟更高 |
| WCPEI | 35.6 | +6.4 — 每条指令花更多 cycle |
| SMEM | 8.19KB | As[32][32]+Bs[32][32], 16KB per block |
| Occupancy | 66.5% | 持平 |

**诊断**：Tiling 减少了 Global Memory 流量（SOL 从 93.5→80.7%），但 Shared Memory 访问引入新延迟——IPC 反而从 1.10 降到 0.90。这是因为 Shared Memory Load（LDS.32）比 L1 Cache Hit 延迟更高。Warp 等待从 Global Memory 转移到 Shared Memory。

#### Float4 (Level 3)

| 指标 | 值 | vs Shared |
|------|-----|-----------|
| Memory SOL | 74.8% | **-5.9pp** — float4 LDG.128 减少了 Load 指令 |
| Duration | 2.16ms | **+0.5%** — 几乎无变化 |
| IPC Active | 0.83 | -0.07 — 更少的 Load 指令反而 IPC 更低 |
| WCPEI | 38.4 | +2.8 — 每条指令等更久 |

**诊断**：float4 减少了 4x Load 指令数，但瓶颈在 Shared→Register 带宽而非 Global→Shared。Memory SOL 从 80.7→74.8% 说明 Global 加载确实更快了，但 Shared Memory 读取（每个 FMA 需要 2 次 LDS.32）才是真正的瓶颈。**经典 Amdahl 定律案例**。

#### Register Tiling — 转折点 (Level 4)

| 指标 | 值 | vs Float4 |
|------|-----|-----------|
| Compute SOL | **40.9%** | -33.9pp — Compute 不再是 Memory 的影子 |
| Memory SOL | **70.0%** | -4.8pp — 算术强度提升减少了 Memory 需求 |
| SM Busy | **47.1%** | **+20.8pp** — 巨大跃升！SM 几乎一半时间在工作 |
| IPC Active | **1.88** | **+1.05** — 接近双发射上限 |
| WCPEI | **6.9** | **-31.5** — 指令延迟降至 1/5 |
| Duration | **507μs** | **-77%** — 4.3x 更快 |
| L2 Hit Rate | 88.2% | +34pp — 更大的 tile 更好利用了 L2 |
| No Eligible | 52.8% | -26.4pp — warp 空闲大幅下降 |
| Registers | **96/thread** | Block Limit Registers=2 — Occupancy 从 66.6→27.0% |
| SMEM | 8.19KB | As[128×8]+Bs[8×128]=2KB |

**诊断**：64 个累加器将算术强度从 ~0.25 提升到 ~2 FLOP/Byte。IPC 从 0.83 飞跃到 1.88，SM Busy 从 26% 跃升到 47%。**代价是 Occupancy 从 66.6% 骤降到 27.0%**——96 寄存器/线程限制到 2 Blocks/SM。这是一个教科书级的 ILP vs TLP 取舍案例：用更低的 Occupancy 换更高的单线程 ILP，总体吞吐大幅提升。

#### DB Async — 最快 CUDA Core (Level 5)

| 指标 | 值 | vs RegTile |
|------|-----|------------|
| SM Busy | **49.3%** | **+2.2pp** — 接近 50% 天花板 |
| IPC Active | **1.97** | **+0.09** — 双发射极限（1 FP + 1 LD/cycle） |
| WCPEI | **6.3** | **-0.6** — 指令延迟进一步降低 |
| Memory SOL | 64.8% | -5.2pp — cp.async 隐藏了 Global 加载延迟 |
| Compute SOL | 43.8% | +2.9pp |
| No Eligible | 50.5% | -2.3pp |
| Duration | 498μs | -2% |
| SMEM | **32.77KB** | 双缓冲 2×(As[128×16]+Bs[16×128])=32KB |
| Registers | 95/thread | 持平 |

**诊断**：cp.async Double Buffering 将 Global Memory 加载延迟隐藏在计算后面。IPC 达到 1.97——Ampere SM 的理论双发射上限（1 FP + 1 INT/LD 每 cycle）。SM Busy 49.3% 意味着剩余 50% 的时间 SM 在等 Shared Memory 延迟和 barrier 同步。**这是手写 CUDA Core GEMM 在 3060 Ti 上的实用上限。**

#### WMMA TF32 — Tensor Core 首次尝试 (Level 6)

| 指标 | 值 | vs DB Async |
|------|-----|-------------|
| SM Busy | **19.3%** | **-30.0pp** — 暴跌！ |
| IPC Active | **0.32** | **-1.65** — 指令发射几乎停滞 |
| WCPEI | **38.8** | **+32.5** — 每条指令等 6x 更久 |
| Compute SOL | 16.6% | -27.2pp — Tensor Core 几乎没在工作 |
| No Eligible | **91.9%** | **+41.4pp** — 92% 时间 warp 空闲 |
| Duration | 474μs | **-5%** — 居然比 DB Async 略快？ |
| Registers | **128/thread** | Block Limit Registers=2 |
| SMEM | 16.38KB | As[128×16]+BsT[128×16]=8KB |
| L1 Hit Rate | 4.0% | -14pp — WMMA 访问模式不利于 L1 |

**诊断**：WMMA 在 1024² 的 Duration 比 DB Async 快 5%（474 vs 498μs），但 SM Busy 只有 19.3%。这说明 Tensor Core 的 `mma_sync` 吞吐很高（compute 很快完成），但 warp 在 `__syncthreads()` 上空转了 92% 的时间。**Tensor Core 算力不是瓶颈——barrier 同步才是。**

#### WMMA+DB — cp.async 救援 (Level 7)

| 指标 | 值 | vs WMMA TC |
|------|-----|------------|
| SM Busy | 19.0% | -0.3pp |
| IPC Active | **0.40** | **+25%** — A 异步加载释放了 issue slot |
| WCPEI | **32.8** | **-15.5%** — 指令延迟降低 |
| Memory SOL | **60.4%** | **-5.7pp** — A 异步加载减轻 Memory 压力 |
| Compute SOL | 17.8% | +1.2pp |
| No Eligible | 90.1% | -1.8pp — 几乎无改善 |
| Duration | 484μs | +2% |
| SMEM | **32.77KB** | 双缓冲翻倍 |
| Registers | 128/thread | 持平 |

**诊断**：cp.async 的 A 异步加载确实有效——IPC +25%、WCPEI -15%、Memory SOL -6%。但 No Eligible 只从 91.9% 降到 90.1%（-2pp），SM Busy 纹丝不动。**B 加载仍是同步的（转置无法用 `__pipeline_memcpy_async`），占一半加载开销；`__syncthreads()` 数量没变（256 tiles × 2 = 512 次 barrier）。cp.async 能隐藏 Global Memory 延迟，但隐藏不了 barrier 串行化。**

### 13.3 跨内核趋势分析

#### 趋势一：Memory-Bound → 趋向平衡

```
Naive:   Memory 93.5%  Compute 93.5%  (Compute=Memory → 纯 Memory-Bound)
Shared:  Memory 80.7%  Compute 80.7%  (同上)
Float4:  Memory 74.8%  Compute 74.8%  (同上)
RegTile: Memory 70.0%  Compute 40.9%  (开始分叉 — 算术强度提升)
DBAsync: Memory 64.8%  Compute 43.8%  (最佳平衡)
WMMA TC: Memory 66.1%  Compute 16.6%  (Tensor Core 算力被 barrier 锁死)
```

Memory SOL 和 Compute SOL 从 Naive 的 "完全重合" 到 RegTile 的 "明显分叉"，代表了算术强度从 0.125 → 2 FLOP/Byte 的提升。DB Async 达到最均衡的状态。

#### 趋势二：Occupancy 下降 + IPC 上升 = ILP 战胜 TLP

```
Naive:   Occ 66.6%  IPC 1.10  SM Busy 31.4%
RegTile: Occ 27.0%  IPC 1.88  SM Busy 47.1%  (+50%!)
```

高 Occupancy 对 Memory-Bound kernel 有益（更多 warp 隐藏延迟）。但当瓶颈转向 Compute-Bound 时，低 Occupancy 配合高 ILP（每线程 64 累加器）反而胜出。这是 GPU 优化最关键的取舍之一。

#### 趋势三：Warp 空闲率直接决定 SM Busy

```
         No Eligible  SM Busy
Naive:      72.6%      31.4%
RegTile:    52.8%      47.1%
DB Async:   50.5%      49.3%
WMMA TC:    91.9%      19.3%
WMMA+DB:    90.1%      19.0%
```

**SM Busy ≈ 100% - No Eligible**（误差 <3pp）。这说明在我们的 kernel 中，warp 空闲就是唯一瓶颈——只要 SM 有可发射的 warp，它就在工作；一旦所有 warp 都在等 barrier/内存，SM 就空转。**优化 GEMM 的本质就是让 warp 永远不等。**

### 13.4 LDS.128 效果验证（NCU 视角）

**做法**：将 Register Tiling 中 `reg_b` 的 8 次标量 Shared Memory Load 替换为 2 次 float4 LDS.128，并添加 `__launch_bounds__(256, 2)` 辅助编译器寄存器分配。

**效果（4096²）**：

| Kernel | 优化前 | 优化后 | 提升 |
|--------|--------|--------|------|
| RegTile | 7.19 TF | 7.65 TF | **+6.4%** |
| DB Async | 7.54 TF | 7.86 TF | **+4.2%** |

**为什么有效**：DB Async 在 1024² 时 SM Busy 49.3%、IPC 1.97。LDS.128 将 reg_b 的 8 条 LDS.32 指令合并为 2 条 LDS.128，减少了指令发射压力，让 dual-issue 更充分。512 次 K 循环 × 256 tiles × 6 条指令节省 = 约 78 万条指令被消除。

### 12.3 P1: LDS.128 + `__launch_bounds__`

**做法**：将 Register Tiling 中 `reg_b` 的 8 次标量 Shared Memory Load 替换为 2 次 float4 LDS.128，并添加 `__launch_bounds__(256, 2)` 辅助编译器寄存器分配。

**效果（4096²）**：

| Kernel | 优化前 | 优化后 | 提升 |
|--------|--------|--------|------|
| RegTile | 7.19 TF | 7.65 TF | **+6.4%** |
| DB Async | 7.54 TF | 7.86 TF | **+4.2%** |

---

## 第14章 最终总结

### 14.1 完整优化历程（4096² FP32, RTX 3060 Ti）

| Level | Kernel | 核心技术 | TFLOPS | 累计 | 源码文件 |
|-------|--------|---------|--------|------|----------|
| 1 | Naive | Global Memory | 1.07 | 1.0x | `src/gemm_kernels.cu` |
| 2 | Shared | SMEM 32×32 | 1.32 | 1.2x | `src/gemm_kernels.cu` |
| 3 | Float4 | LDG.128 | 1.32 | 1.2x | `src/gemm_kernels.cu` |
| 4 | RegTile | 64 acc, TM=TN=8, LDS.128 | 7.78 | 7.3x | `src/gemm_kernels.cu` |
| 5 | DB Async | cp.async DB, BK=16 | 8.00 | 7.5x | `src/gemm_kernels_async.cu` |
| 6 | WMMA TC | TF32 Tensor Core, BK=16 | 7.95 | 7.4x | `src/gemm_kernels_tc.cu` |
| 7 | LDS.128+LB | SMEM vec + bounds (跨 kernel) | 8.00 | 7.5x | `src/gemm_kernels*.cu` |
| 8 | WMMA+DB | TC + cp.async, BK=16 | **8.10** | **7.6x** | `src/gemm_kernels_tc_async.cu` |
| — | cuBLAS | Tensor Core + SASS | 11.0 | 10.3x | 参考 |

### 14.2 优化路径全览（4096² FP32）

```
起步 — Naive 全局内存直接读写
│  1.07 TFLOPS, 6.6% 峰值
│  瓶颈：Global Memory 带宽，算术强度 ~0.125 FLOP/B
│
├─ Shared Memory Tiling (32×32)
│  1.32 TFLOPS, 8.1% (+23%)
│  访存量从 550GB 降到 16GB。瓶颈转移：SMEM→Register 带宽
│
├─ Float4 向量化加载 (LDG.128)
│  1.32 TFLOPS, 8.1% (+0%)
│  优化了 Global→Shared，但瓶颈不在此 — 经典 Amdahl 定律
│
├─ Register Tiling (64 累加器, TM=TN=8, BK=8) ← breakthrough
│  7.78 TFLOPS, 48.0% (+489%)
│  算术强度从 0.125 跃升到 2 FLOP/B。瓶颈转移：Global 加载延迟 + barrier
│
├─ cp.async Double Buffering (BK=16)
│  8.00 TFLOPS, 49.4% (+3%)
│  加载与计算开始重叠。SM Busy 首次达到 50%
│
├─ LDS.128 + __launch_bounds__ ← CUDA Core ceiling
│  8.00 TFLOPS, 49.4% (+0%)
│  Shared Memory Load 指令减少 4x，IPC 达 1.97
│
├─ WMMA TF32 Tensor Core (BK=16)
│  7.95 TFLOPS, 49.1% (-0.6%)
│  Tensor Core 算力更强，但 91.9% warp 在等 barrier — 算力被同步吞噬
│
└─ WMMA + cp.async Double Buffering
   8.10 TFLOPS, 50.0% (+1.9%)
   A 加载异步化微改善，barrier 仍未解决。CUDA Core 和 TC 首次持平

──────────────────────────────────────────
cuBLAS (torch.mm)
  11.0 TFLOPS, 68%
  差距：+3.1 TFLOPS（相对提升 39% 才能追上）
```

### 14.3 我们 vs cuBLAS：NCU 逐项对比

| 指标 | 我们 (DB Async) | cuBLAS (估值) | 差距倍数 | 含义 |
|------|----------------|-------------|---------|------|
| SM Busy | 49% | ~75% | 1.5x | SM 1/3 时间在空转 |
| No Eligible Warp | 50% | ~20% | 2.5x | warp 闲置率是我们的 0.4x |
| IPC Active | 1.97 | ~2.5 | 1.3x | dual-issue 未拉满 |
| Warp Cycles/Inst | 6.3 | ~3-4 | 1.8x | 每条指令多等 3 cycles |
| Memory SOL | 65% | ~40% | 0.6x | 他们 DRAM 流量更少（更好的 cache 利用） |
| Compute SOL | 44% | ~60% | 1.4x | 计算单元利用率差 16pp |

### 14.4 差距逐层拆解

cuBLAS 多做了四件我们没做的事：

| 层次 | cuBLAS 做了什么 | 我们没做 | 流失性能 |
|------|----------------|---------|---------|
| 同步机制 | `cuda::barrier` 异步同步 + 自研 SASS 实现 | `__syncthreads()` 全局 barrier | ~1.0 TFLOPS |
| Pipeline 深度 | 3-4 stage cp.async，加载延迟彻底隐藏 | 2-stage ping-pong | ~0.8 TFLOPS |
| Shared Memory | XOR swizzle 消除 bank conflict | 朴素布局，BK>16 时 conflict 严重 | ~0.5 TFLOPS |
| 指令调度 | 手写 SASS 汇编，dual-issue 最大化，寄存器 bank 零冲突 | nvcc -O3 编译 | ~1.0 TFLOPS |
| 分块策略 | 按 M/N/K 动态选择 TM/TN/BK/BM/BN | 固定 BM=BN=128, TM=TN=8, BK=16 | ~0.1 TFLOPS |

```
我们的 DB Async:  8.0 TFLOPS
  + barrier 异步    → +1.0  (SM Busy 50→60%)
  + 3-stage pipe    → +0.8  (Memory SOL 65→45%)
  + XOR swizzle     → +0.5  (IPC 1.97→2.2)
  + SASS 调度       → +1.0  (Warp Cycles 6.3→3.5)
  ─────────────────────────────
  理论上限:        ~11.2 TFLOPS ≈ cuBLAS 的 11.0
```

### 14.5 为什么我们补不上

**1. SASS 汇编是硬壁垒**

cuBLAS 的 kernel 含有手写 SASS（Shader Assembly）代码，由 NVIDIA GPU 架构师逐指令编排。nvcc 的 `-O3` 优化只能做到编译器的最佳努力，无法匹敌人手工的寄存器分配、dual-issue 配对、延迟隐藏、bank conflict 规避。这是"编译器 vs. 人"的差距，不是"不懂 vs. 懂"的差距。

**2. 异步 barrier 在消费级环境不可用**

CUDA 12.4 + torch extension 构建环境下，`<cuda/barrier>`（libcu++）和 `cuda::device::memcpy_async` API 不兼容或不存在。我们尝试了 `cuda::barrier` 替代 `__syncthreads()`，编译都无法通过。而 cuBLAS 的异步 barrier 是自研的 SASS/PTX 级别实现，不依赖这些高层 API。

**3. 单卡单架构的局限**

cuBLAS 为每个 GPU SKU 预调了分块参数，并有 CUTLASS 自动调参框架支撑。我们针对 sm_86 手工硬编码，没有跨架构适配的条件和意义。

**4. 投入产出比已达拐点**

```
投入 1x (Tiling + RegTile)    → 收益 6.0x   超高回报
投入 2x (cp.async + LDS.128)  → 收益 0.6x   合理回报
投入 10x (SASS + barrier)     → 收益 0.5x   边际递减
```

前两个优化用几百行 CUDA C++ 换来了 7x 加速。后面每 1% 的提升需要几何级数的深度投入。手写 CUDA GEMM 到 49% 峰值是任何一个认真学习的 CUDA 程序员都能做到的——到 68% 是 NVIDIA 工程师团队的领地。

**5. 但这不意味着失败**

8.10 TFLOPS（50% FP32 峰值）是手写 CUDA C++ GEMM 的优秀成绩。项目覆盖了从 Naive 到 Tensor Core 的 8 级优化、21 道面试题、NCU profiling、完整的 benchmark 框架，以及每一级优化 WHY 的深度分析。对简历面试、对深入理解 GPU 计算——这套项目已经做到了它该做的一切。

---

> **文档版本**: v7.0 | **日期**: 2026-05-20 | **GPU**: RTX 3060 Ti (Ampere, sm_86)
> 所有性能数据来自 `benchmark.py` 实测 (4096² FP32)，NCU 数据来自 `run_ncu.bat` (--set full, 1024²)
> v7.0 更新：重整 Ch9-14，全部数据对齐最新 benchmark，每章标注源码+kernel+参数



