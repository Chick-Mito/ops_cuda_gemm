# 📌 学习笔记：Naive GEMM 核心计算流与底层调度机制

## 一、naive版本性能分析

### 0.代码

```cpp
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

```



### 1. 宏观空间切分（Grid & Block 级别）

在进行 $M \times N \times K$ 的大矩阵乘法（如 $4096 \times 4096$）时，GPU 无法一次性处理如此庞大的数据。因此，程序在逻辑上将结果矩阵 $C$ 划分为二维的分块网格。

- **分块配置**：设置线程块大小 `dim3 block(32, 32);`（即 `BLOCK_SIZE = 32`）。
- **网格划分**：整个结果矩阵 $C$ 被切分为 $128 \times 128 = 16,384$ 个独立的小方块。每个小方块对应一个 **Thread Block（线程块）**。
- **硬件派发**：这 16,384 个 Block 作为独立的任务单元，由 GPU 架构调度器动态分发到显卡芯片的各个 **SM（流式多处理器）** 上排队执行。块与块之间完全并行、互不干扰。

### 2. 微观线程映射（Block & Warp 级别）

进入具体的 SM 后，一个负责 $32 \times 32$ 局部矩阵 $C$ 的 Block 开始执行。其内部拥有 $32 \times 32 = 1024$ 个线程。

- **一维线性化（Linearization）**：GPU 硬件无法直接识别二维坐标，它会通过下式将二维线程打平成一维的绝对序号 `tid`：

  $$tid = threadIdx.y \times 32 + threadIdx.x$$

- **Warp 划分（线程束绑定）**：硬件规定**每连续 32 个 `tid` 绑定为一个物理执行单位 —— Warp（线程束）**。由于 `x` 维度（对应矩阵的列 `col`）变动最快，产生了一个完美的空间映射：

  - **Warp 0** (`tid: 0~31`)：它们的 `threadIdx.y` 全都为 0。意味着 Warp 0 承包了该 Block 负责区域的**第 0 行（包含连续 32 个元素）**。
  - **Warp 1** (`tid: 32~63`)：它们的 `threadIdx.y` 全都为 1。承包了该 Block 负责区域的**第 1 行**。
  - **Warp 31**：承包了该 Block 负责区域的**第 31 行**。

- 32 个 Warp 在空间上横向拉开，纵向堆叠，刚好拼出了整个 Block 的 $32 \times 32$ 方块。

### 3. 时间轴纵深推进（内层循环 $K$ 轴）

为了求出结果矩阵 $C$ 中的任何一个元素，在数学上必须完整履行 $K$ 次乘加操作（向量内积）。以负责计算 $C[\text{row}][0 \sim 31]$ 的 **Warp 0** 为例，其内部 32 个线程在 **SIMT（单指令多线程）** 机制下齐步走，展开 `for(int i = 0; i < K; i++)` 的纵深推进：

- **第 `i = 0` 步**：32 个线程同时发起对 Global Memory 的访问。
  - **矩阵 A（广播访存 Broadcast）**：32 个线程的 `row` 相同，`i` 相同。硬件触发广播机制，仅从显存拉取一次 $A[\text{row}][0]$，分发给全员。
  - **矩阵 B（合并访存 Coalesced）**：32 个线程的 `i` 相同（都在 B 的第 0 行），而 `col` 为连续的 `0 ~ 31`。地址在物理上是 128 字节连续的。硬件触发合并访存，单次内存事务打包抓取 $B[0][0 \sim 31]$。
  - **计算（FMA）**：32 个 CUDA Core 同时发力，做乘加操作，结果累加到各线程私有的寄存器 `sum` 中。
- **第 `i = 1` 步**：全员迈向下一步，广播读取 $A[\text{row}][1]$，合并读取 $B[1][0 \sim 31]$，持续累加。
- **第 `i = K-1` 步**：经历 $K$ 轮循环后，各线程的寄存器 `sum` 中终于存下了完美的数学解。Warp 内 32 个线程同时执行 `C[row * N + col] = sum;`，**合并地**写回全局内存。

### 4. 痛点剖析：为什么 Naive 模式性能极其惨淡？

虽然 Naive 模式在 Warp 内部做到了完美的 **Broadcast（读A）** 和 **Coalesced（读B）**，但它在 RTX 3060 Ti 上通常只能跑出 **1.0 TFLOPS** 左右的低效算力。其根本原因并非访存不合并，而是遭遇了严重的 **Memory Bound（访存瓶颈导致的算力饥饿）**。

#### ① 灾难级的数据复用率（Data Reuse = 0）

每个线程完全独立地去全局内存拉取数据。在 Block 内部，Warp 0 读过了 $B[0][0 \sim 31]$，接下来的 Warp 1、Warp 2... 直到 Warp 31 也要算自己的行，它们**不得不重新去极慢的全局内存中拉取一模一样的 $B[0][0 \sim 31]$**。大矩阵的每一个元素都被反复读取了几千次，海量的重复访存请求瞬间挤爆了显存总线带宽。

#### ② 隐藏延迟失败与算力饥饿

GPU 依赖高并发来隐藏访存延迟。当 Warp 0 发出访存请求时，由于去显存拿数据需要 200~400 个时钟周期，Warp 0 被挂起（Stall）。调度器零开销切换到 Warp 1，但 Warp 1 也没有现成数据，同样伸手要数据并挂起。

由于整个 Kernel 充斥着毫无复用的 Global Memory 访问，瞬间**全片显卡上的所有 Warp 都会因为等待数据而全部卡死挂起**。Warp 调度器的就绪队列直接清空，导致高算力的 CUDA Cores 在绝大多数时间里都在空转干等。

### 💡 优化的必由之路

理解了 Naive 的死穴在于“没有公共大水缸（缓存共享），导致显存带宽被打满”，后续的优化方向便完全明朗：

- **Level 2 (Shared Memory)**：引入片上高速共享内存，让 Block 内的线程协力把数据搬到“公共大水缸”里，内部共享，将全局内存的访存频次暴降 `BLOCK_SIZE` 倍。
- **Level 4 (Register Tiling)**：让每个线程不再只算 1 个格子，而是通过寄存器分块一口气算 $8 \times 8$ 个格子，将片上共享内存的访存压力再降低 8 倍，彻底释放硬件算力。

## 二、shared memory版本优化分析

### 0. 代码

```cpp
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
```



现在我们正式跨入矩阵优化的 **Level 2：共享内存分块（Shared Memory Tiling）**。

在 Naive 模式中，我们看到了一个极其低效的现象：同一个 Block 内的 32 个 Warp，在沿着 $K$ 轴挑水（求内积）时，明明大家需要的 $B$ 矩阵数据高度重合，却各自为战，成百上千次地去撞击极慢的 Global Memory。

Level 2 的核心思想，就是**在 SM（流式多处理器）内部建立一个“片上公共大水缸”——共享内存（Shared Memory, SMEM）**。

### 1.  核心策略：化纵为横，分段拦截

在 Naive 模式下，线程是一口气沿着 $K$ 轴从 `0` 一直走到 `4096`。

在 Level 2 模式下，我们把漫长的 $K$ 轴切成一段一段的**方块（Tile）**。因为我们的 `BLOCK_SIZE = 32`，所以最自然的切法就是把 $K$ 轴切成大小为 32 的小段。

大矩阵乘法被重构为：**整个 Block 内的 1024 个线程先停下手里算内积的工作，大家齐心协力，先把 $A$ 矩阵的 $32 \times 32$ 小方块和 $B$ 矩阵的 $32 \times 32$ 小方块，从 Global Memory 搬到片上的共享内存中。搬完之后，大家再在这个高速的“片上大水缸”里疯狂复用数据，算完这一段，再迈向下一段。**

我们在代码里声明的这两个静态数组，就是片上大水缸：

C++

```
__shared__ float As[BLOCK_SIZE][BLOCK_SIZE]; // 32x32 的片上高速 A 缓存
__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE]; // 32x32 的片上高速 B 缓存
```

*(注：共享内存是隐藏在 GPU 芯片内部的 SRAM，其访存延迟通常只有几个时钟周期，比外面的 Global Memory 快几百倍！)*

### 2.  源码级拆解：一个 Tile 的生命周期

我们引入最外层的分段循环 `for(int tile = 0; tile < K / 32; tile++)`。在某一个具体的 `tile` 循环内，Block 内部的 1024 个线程会经历以下三个严格的步骤：

#### 阶段 1：协力挑水，填满水缸（Cooperative Loading）

这个 Block 内部有 1024 个线程，而我们要搬运的 $As$ 矩阵有 $32 \times 32 = 1024$ 个格子，$Bs$ 矩阵也有 1024 个格子。

**分工极其明确：每个线程负责搬运 $A$ 的 1 个元素和 $B$ 的 1 个元素。**

以任意一个线程 `(threadIdx.y, threadIdx.x)` 为例：

C++

```
// 每个线程去 Global Memory 对应的物理格子上舀一瓢水
As[threadIdx.y][threadIdx.x] = A[global_row * K + (tile * 32 + threadIdx.x)];
Bs[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * N + global_col];
```

1024 个线程同时出手，瞬间就从 Global Memory 中抓取了 1024 个 $A$ 的元素和 1024 个 $B$ 的元素，并整整齐齐地填进了片上的 $As$ 和 $Bs$ 共享内存中。

#### 阶段 2：强制集合，安全哨兵（__syncthreads）

数据刚往共享内存里写，由于不同的 Warp 跑得有快有慢，可能 Warp 0 已经把它的那部分写完了，而 Warp 5 还在路上。如果此时立刻开始计算，Warp 0 就会读到 Warp 5 还没来得及写入的垃圾数据。

于是，代码里出现了一条至关重要的物理铁律：

```cpp
__syncthreads(); // Block 级别的同步栅栏
```

这是一道命令：**这个 Block 内部的所有 1024 个线程，不管你跑得多快，到这里必须全部踩刹车停下！** 只有当最后一个人把数据写进共享内存、整个 $32 \times 32$ 的公共大水缸彻底被填满填正确之后，栅栏才会打开，全员同时放行。

#### 阶段 3：水缸泡茶，极限复用（Compute inside SMEM）

大水缸里的水现在是安全的了。接下来，32 个 Warp（1024个线程）开始算它们各自的中间值。

依然以负责 $C$ 矩阵第 0 行的 **Warp 0** 为例，它的 32 个线程要在局部的高速水缸里走一个 32 次的小循环：

C++

```
for(int k = 0; k < 32; ++k) {
    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // 全部在超高速的片上 SRAM 里完成
}
```

看清这里发生的**恐怖复用**了吗？

- Warp 0 在计算 `k=0` 时，读取了 `As[0][0]`。
- 下一时刻，Warp 1 算它的行，也跑来读取 `As[1][0]`。
- 最关键的是，**Bs[0][threadIdx.x] 这 32 个数，被这个 Block 内部整整 32 个 Warp、1024 个线程反复、同时、疯狂地读取了 32 次！**
- 这些读取全部发生在片上，根本不需要惊动外面的显存。

算完这 32 次之后，大家手里累加好局部结果。接着再调用一次 `__syncthreads();`，确保所有人都算完了，然后再迈入下一个 `tile`，去洗净大水缸，搬运下一段的 $32 \times 32$ 数据。

### 3.  为什么能将 Global Memory 访存频次暴降 BLOCK_SIZE 倍？

💡 访存定量分析（以单个 Block 计算 32×32 区域、矩阵规模 4096² 为例）

我们来做一道极其震撼的数学对比题。假设我们要算出这个 Block 对应的 $32 \times 32 = 1024$ 个结果元素，沿着 $K=4096$ 的轴：

#### 1. Naive 模式下的 Global Memory 访问次数：

- **单个线程工作量**：需独立纵深推进 $K$ 轴，读取 $A$ 的一行（4096）与 $B$ 的一列（4096），共访存 $4096 \times 2 = 8192$ 次。
- **单个 Block（1024线程）总计**：$1024 \times 8192 = 8,388,608$ 次。

即：

- 一个 Thread：2K 次。
- 一个 Block：32×32×2K = 2048 K 次

#### 2. Level 2 (Shared Memory) 模式下的 Global Memory 访问次数：

- **切分逻辑**：将 $K$ 轴切分为 $4096 / 32 = 128$ 个 Tile。
- **单个 Tile 内**：Block 内 1024 个线程协同，整体只搬运 $32\times32$ 的 $A$ 块与 $B$ 块。此时单个 Block 对 Global Memory 访存为 $1024 + 1024 = 2048$ 次（均摊到每个线程仅访存 2 次）。
- **单个 Block 纵深推进 128 个 Tile 总计**：$128 \text{ Tiles} \times 2048 \text{ 次} = 262,144$ 次（均摊到每个线程在整个生命周期仅访存 $128 \times 2 = 256$ 次）。

即：

+ 一个 Thread：2次
+ 一个 Block：32×32×2 = 2048 次。分了 K/block_size = K/32 个块
+ 共：2048 K / 32 次

$$\text{访存削减比例} = \frac{8,388,608}{262,144} = 32 \text{ 倍} \quad (\text{正好等于 } BLOCK\_SIZE)$$

**结论：通过引入片上共享内存，我们成功把从“地狱般缓慢”的 Global Memory 读取数据的次数，整整砍掉了 32 倍！**

### 4. 遗留的瓶颈：为什么性能只从 1.0 提升到了 1.3 TFLOPS？

既然访存砍掉了 32 倍，为什么 RTX 3060 Ti 的表格里，性能提升却如此微弱（只到 1.3 TFLOPS）？

因为我们**刚刚打破了全局内存墙，就迎面撞上了共享内存墙。**

在这段内层计算代码里：

C++

```
sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
```

每一个线程每进行一次乘加（FMA）指令，都需要发射**两条从 Shared Memory 读取数据的指令**。

1024 个线程在并发执行时，会产生极其密集的 Shared Memory 访存请求。尽管 Shared Memory 很快，但它的**指令发射总线带宽和硬件 Bank 冲突（Bank Conflict）** 瞬间被拉到了极限。

也就是说，此时 GPU 的计算单元依然在原地踏步，它们只是从“等显存数据”变成了“等共享内存数据”。这就逼着我们必须迈向 **Level 4（Register Tiling）**——让每个线程把数据从共享内存进一步拉到自己专属的寄存器（Register）里，在更高级的寄存器内部完成 $8 \times 8$ 的计算，从而把共享内存的压力再打掉 8 倍，实现 7.2 TFLOPS 的惊天暴涨。

## 三、float4 向量化读取版本优化分析

### 0. 代码

```cpp
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

```



从 Level 2 到 Level 4 的这段演进，是整个 GPU 算子优化中最精彩、最能体现“榨干硬件”思维的阶段。

我们要回答的，本质上是 GPU 芯片内部的**两条总线颈瓶切换**的问题：**为什么 Float4（Level 3）没有打破瓶颈，而 Register Tiling（Level 4）却能让性能暴涨 5.5 倍？**

### 1.  为什么 Level 3 (Float4) 性能完全没有提升？

在 Level 2 中，我们通过共享内存（Shared Memory）把全局内存（Global Memory）的拉取次数砍掉了 32 倍。

这时候，引入了 `float4` 向量化访存：

```cpp
// 每次不再读 1 个 float，而是一口气读 1 个 float4（4个float，128 bits）
float4 a_vec = *A_ptr;
As[load_r][load_c + 0] = a_vec.x;
As[load_r][load_c + 1] = a_vec.y; // ... 搬运到共享内存
```

- **它的实际作用**：它极大地优化了“从 Global Memory $\rightarrow$ 到 Shared Memory”的这一段搬运效率（触发了底层的 `LDG.128` 硬件指令，让总线每次吃满 128 位，减少了指令发射数）。

- **为什么没效果（表格里依然是 1.3 TFLOPS）**：

  因为在 Level 2 的时候，**“从 Global Memory 搬到 Shared Memory”这件事情，已经不是最核心的瓶颈了！** 真正的瓶颈卡在**内层计算循环**里。

我们看看内层的计算代码：

```cpp
for(int k=0; k<BLOCK_SIZE; ++k) {
    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // 核心死穴在这里！
}
```

在这个循环里，每个线程为了做**1 次**乘加计算（FMA），必须发射**2 次**从 Shared Memory 读取数据的指令（读一次 `As`，读一次 `Bs`）。

这导致了两个极其致命的后果：

1. **共享内存带宽被打满**：虽然 Shared Memory 比 Global Memory 快几百倍，但它依然是有带宽极限的。1024 个线程疯狂地发射物理读指令，直接让 Shared Memory 的数据总线彻底瘫痪。
2. **严重的硬件 Bank 冲突（Bank Conflict）**：多个线程同时访问同一个 Bank 的不同地址时，请求会被强制串行化，导致严重的流水线等待。

**总结**：`float4` 只是优化了“进货”（搬运数据到水缸）的效率。但现在卡死显卡的是“漏斗太小”（从水缸里频繁捞水做计算的开销太大）。货进得再快，计算单元吃不进去，性能自然纹丝不动。



## 四、 register tiling 版本优化分析

### 0. 代码

```cpp
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

    float accum[TM][TN] = {0.0f}; // 线程私有的 8x8 结果寄存器
    float reg_a[TM] = {0.0f};     // 线程私有的 A 矩阵数据寄存器
    float reg_b[TN] = {0.0f};     // 线程私有的 B 矩阵数据寄存器

    // ... (索引计算部分省略，详见源码) ...

    for (int tile = 0; tile < (K + BK - 1) / BK; ++tile) {
        // 1. Float4 协同搬运 Global -> Shared Memory (同 Level 3)
        // ...
        __syncthreads();

        // 2. Register-level compute (核心爆改区域)
        for (int k = 0; k < BK; ++k) {
            // 从 Shared Memory 提水到私有寄存器 reg_a
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[(thread_r + i) * BK + k]; 
            }
            // 从 Shared Memory 提水到私有寄存器 reg_b (使用 float4 向量化指令 LDS.128)
            float4 vb0 = *reinterpret_cast<float4*>(&Bs[k * BN + (thread_c + 0)]);
            float4 vb1 = *reinterpret_cast<float4*>(&Bs[k * BN + (thread_c + 4)]);
            reg_b[0] = vb0.x; reg_b[1] = vb0.y; reg_b[2] = vb0.z; reg_b[3] = vb0.w;
            reg_b[4] = vb1.x; reg_b[5] = vb1.y; reg_b[6] = vb1.z; reg_b[7] = vb1.w;

            // 3. 在极速的寄存器内部完成 8x8 的外积乘加 (FMA)
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    // 4. 将 8x8 的结果从寄存器写回 Global Memory
    // ...
}
```

### 1. 核心思想：建立“私有水杯”，降维打击

在 Level 2/3 中，卡死算力的是“共享内存（Shared Memory）带宽墙”——做 1 次乘法，就要发射 2 条指令去 Shared Memory（片上水缸）捞数据。

Level 4 的破局之道，是在 Shared Memory 和 CUDA Core 之间，再插入一层速度最快、延迟几乎为 0 的存储：**寄存器（Registers）**。通过引入 **Register Tiling（寄存器分块）**，我们彻底改变了线程的“打工模式”：

- **以前（Level 1/2/3）：** 1 个线程只负责计算结果矩阵 C 中的 **1 个点**。
- **现在（Level 4）：** 1 个线程一口气吞下 64 个工作量，独占结果矩阵 C 中的一个 **$8 \times 8$ 的子矩阵块（Tile）**！

代码开头声明的 `accum[8][8]`、`reg_a[8]`、`reg_b[8]` 就是给每个线程分配的“私有水杯”。它们会被编译器直接映射到 GPU 极其珍贵的物理寄存器上。

### 2. 宏观宏图：大矩阵是如何被一步步吞掉的？

假设我们要计算 $M \times N \times K = 4096 \times 4096 \times 4096$ 的巨型矩阵乘法，整个硬件层面的并行动作可以分为三个阶段：

#### 阶段一：空间排兵布阵（Grid $\rightarrow$ Block $\rightarrow$ Thread）

1. **宏观切块（Grid Level）：** 结果矩阵 C（$4096 \times 4096$）被切成了许多个 $128 \times 128$ 的大方块。每个方块由一个 Block 包干，总共需要 $\frac{4096}{128} \times \frac{4096}{128} = 1024$ 个 Block。它们作为独立任务被分发到各个 SM 上。
2. **微观分工（Block Level）：** 进入某个 Block 后，面对 $128 \times 128$ 的领地，它派出了 $16 \times 16 = 256$ 个线程（Thread）。每个线程按片包干，各自领走一个 **$8 \times 8$ 的终极小领地**。

> **💡 破除疑问 1：为什么我的 accum 只有 $8 \times 8$？**
>
> 因为你写的 `float accum[8][8]` 只是**单个线程**的私有存钱罐。就在此时此刻，整张显卡上有 $1024 \times 256 = 262,144$ 个线程在同时开火，每个人手握一个 $8 \times 8$ 的杯子，拼起来就是整个超大矩阵。

#### 阶段二：时间纵深推进（沿着 K 轴分批冲锋）

空间领地（$8 \times 8$）定死后，线程需要合力沿着长达 4096 的 K 轴长途跋涉。因为片上内存极其昂贵，必须分批推进，这就是外层大循环 `for (int tile = 0; ...)`。这里 `#define BK 8`，意味着把 4096 长的 K 轴切成了 $\frac{4096}{8} = 512$ 个批次（Tiles）。

以第 0 个 Tile（$k=0 \sim 7$）为例，其微观执行流如下：

1. **全员进货（Global $\rightarrow$ Shared）：** 256 个线程暂时放下算盘化身搬运工。利用 Level 3 留下的 **float4 向量化显存字指令**，合力将 A 的 $128 \times 8$ 局部块和 B 的 $8 \times 128$ 局部块，批发进高速公共水缸 `As` 和 `Bs` 中。

2. **物理大闸门（`__syncthreads()`）：** 所有人必须在此停下！确保一整车砖全卸进共享内存后，闸门才打开，全员同时进入计算状态。

3. **外积魔法与寄存器复用（Shared $\rightarrow$ Register $\rightarrow$ 计算）：**

   进入内层循环 `for (int k = 0; k < 8; ++k)`。在任意一个具体的 $k$ 步中：

   - **提水进杯：** 线程沿着 `As` 水缸的第 $k$ 列，垂直抠出连续的 8 个元素放到自己的 `reg_a`（列向量）；同时利用 `float4` 地址强转，**触发底层 LDS.128 指令**，从 `Bs` 水缸的第 $k$ 行一口气抠出 8 个元素放到 `reg_b`（行向量）。原本需要 8 次访存指令，现在只需 2 次 LDS.128 就填满了寄存器，极大地缓解了指令发射器的压力。

   - **纯寄存器外积：** 线程在毫无延迟的私有寄存器内部，让这 8 个 A 和 8 个 B 交叉相乘：

     $$\text{列向量 } A_{8 \times 1} \times \text{行向量 } B_{1 \times 8} = \text{局部矩阵 } C_{8 \times 8}$$

     代码中的双重循环执行了 **$8 \times 8 = 64$ 次乘加（FMA）** 操作。

> **💡 破除疑问 2：答案为什么是对的？不是只有局部结果吗？**
>
> 关键就在于 `accum` 声明在 tile 循环的**外面**。在整个长达 512 次的大循环里，`accum` 从来没有被清零过。它像一个存钱罐，通过 `+=` 默默地把 512 个批次里的局部外积结果一路死磕地累加起来。当 K 轴走完时，里面装的绝对是一个完整的最终数学解。

#### 阶段三：衣锦还乡（写回 Global Memory）

K 轴长途跋涉结束，大循环退出。256 个线程手里的 `accum[8][8]` 已经攒满了最终答案。每个线程最后一次出手，将这 64 个寄存器里的终极结果，整整齐齐地写回到最外层慢速的显存矩阵 C 中。





### 3. 访存定量分析：为什么能暴涨 5.5 倍？

为了看清 Level 4 的威力，我们来做一次精密的定量计算。**同样是计算 $8 \times 8 = 64$ 个 C 矩阵的元素：**

- **Level 2/3 (未引入寄存器分块)：**

  需要 64 个线程。它们每做 1 次乘加，都需要向 Shared Memory 发起 2 次读请求。

  $$\text{Shared 读请求次数} = 64 \times 2 = 128 \text{ 次}$$

  $$\text{访存/计算比} = \frac{128}{64} = 2.0$$

- **Level 4 (Register Tiling)：**

  只需要 1 个线程。它只向 Shared Memory 读取了 8 个 A 和 8 个 B。

  $$\text{Shared 读请求次数} = 8 + 8 = 16 \text{ 次}$$

  $$\text{访存/计算比} = \frac{16}{64} = 0.25$$

**结论：** 通过让 1 个线程扛起 $8 \times 8$ 的任务，我们在数学机制上**将 Shared Memory 的读取次数整整削减了 8 倍（128 / 16）**！硬件总线瞬间畅通无阻，这就是算力能飙升到 7.2 TFLOPS 的根本原因。

### 4. 细节狂魔：Float4 的终极归宿 (`LDS.128`)

还记得 Level 3 中无用武之地的 `float4` 吗？在 Level 4 中它终于大放异彩，但这不仅是在 Global -> Shared 的搬运上，更是在 **Shared -> Register 的搬运上**！

看这句极致硬核的代码：

```cpp
float4 vb0 = *reinterpret_cast<float4*>(&Bs[k * BN + (thread_c + 0)]);
```

当线程需要从 Shared Memory 提取 8 个 B 元素到 `reg_b` 时，它没有用一个 `for` 循环读 8 次。而是直接将 Shared Memory 的地址强转为 `float4*`，一口气读取 4 个 float。

- 物理层面：触发了底层的 `LDS.128` (Load Shared 128-bit) 机器指令。
- 作用：原本需要 8 条访存指令，现在只需要 **2 条** `LDS.128` 指令就能填满 `reg_b[8]`！极大地减少了指令发射器 (Instruction Fetcher) 的压力，进一步隐藏了延迟。

### 5. 寄存器个数限制

> 之前的kernel不是一个block是32×32吗，到这里怎么变了：它派出了 $16 \times 16 = 256$ 个 Thread（线程）。

你发现了这套算子优化中最精妙的一个数字解耦！你的敏感度非常高，这个变化绝非拍脑袋决定的，而是**因为我们在 Level 4 改变了“人均工作量（线程计算粒度）”**。

简单直接地回答：**之前的版本是一个线程只算 1 个点，而到了 Level 4，一个线程一口气要算 8×8=64 个点。工作量变大了，完成同样的宏观任务，需要的兵力（线程数）自然就变少了。**

我们通过数学账本和硬件底层限制（寄存器压力）来彻底说清楚为什么是 16×16。

#### 1. 数学账本的对齐：工作量守恒定理

我们来看两个版本中，**一个 Block 负责的总面积**与**线程形状**是如何完美扣合的：

**🔄 之前的版本 (V1 / V2 / V3)**

- **线程工作量**：每个线程只负责计算结果矩阵 *C* 中的 **1 个元素**。

- **Block 的宏观目标**：由于 `#define BLOCK_SIZE 32`，一个 Block 负责计算 32×32=1024 个 *C* 的元素。

- **兵力计算**：

  线程数=单线程工作量Block 负责的总面积=1×132×32=32×32=1024 个线程

**🚀 现在的版本 (Level 4 Register Tiling)**

- **线程工作量**：引入了寄存器分块，由于 `#define TM 8` 和 `#define TN 8`，一个线程要负责计算结果矩阵 *C* 中的一个 **8×8=64 的小矩阵块**。

- **Block 的宏观目标**：在代码开头，宏定义了 `#define BM 128` 和 `#define BN 128`，这意味着一个 Block 负责的目标面积扩大了，变成了 **128×128** 的区域。

- **兵力计算**： 既然一块地是 128×128，每个人能包干 8×8，那我们在纵向和横向分别需要多少人？

  - 纵向（Y轴）需要：128/8=16 个人
  - 横向（X轴）需要：128/8=16 个人

  所以，Block 的形状被严丝合缝地定义为了 `dim3 block(16, 16);`，总线程数就是 16×16=256 个。

#### 2. 硬件层面的终极权衡：惊人的“寄存器暴涨”

你可能会问：“既然要把 Block 负责的面积搞大，为什么不保持 32×32=1024 个线程，且让每个人依然算 8×8 呢？”

如果真的这么做，那就是**硬件上的自杀行为**。因为这会触发 GPU 优化中另一个恐怖的杀手——**寄存器压力（Register Pressure）与 Occupancy（入驻率）暴跌**。

在 GPU 中，寄存器是极其珍贵且数量有限的硬件资源（每个 SM 顶多几万个）：

- 在 Level 4 中，一个线程由于要算 8×8 的外积，它必须在代码里声明 `float accum[8][8]` (64个)、`float reg_a[8]` (8个)、`float reg_b[8]` (8个)。再加上一些索引控制变量，**一个线程足足要吃掉 80 多个硬件寄存器**！
- **如果保持 1024 个线程**：一个 Block 就会吃掉 1024×80≈81,920 个寄存器。这直接超过了绝大多数 GPU 单个 SM 拥有的寄存器物理上限！编译器会因为无片上空间可用，被迫把寄存器里的数据吐到极慢的显存里（称为 Register Spilling），性能会瞬间拉稀。
- **缩减到 256 个线程**：一个 Block 消耗 256×80≈20,480 个寄存器。这个数字非常健康，操作系统可以在一个物理 SM 里同时塞进去 2 个或更多的 Block 一起跑，实现了完美的硬件隐藏延迟。

在代码中，有一句非常硬核的编译器编译器指示：

```cpp
__launch_bounds__(256, 2)
```

这句代码就是显式地告诉 GPU 编译器：“我的这个 Kernel，每个 Block **死定为 256 个线程**，并且请你在硬件优化上，确保一个 SM 里面能**同时塞进至少 2 个 Block** 开火！”

#### 总结

从 32×32 变成 16×16，是**算法数学层面的分工重构**与**硬件底层资源限制**相互妥协、完美解耦后的终极艺术结果：

1. **战术变了**：从“千人规模、每人干苦力算 1 点”的底层模式，升级为了“精英小队、每人狂轰 64 点外积”的高效模式。
2. **数量降了**：线程数从 1024 降到 256，但因为每个人手里有寄存器大杀器，整个 Block 吞掉的矩阵面积反而从 32×32 扩大到了 128×128！



### 📌 总结：矩阵乘法的“三级火箭”

GPU 优化的本质，就是抗击由于“算存不平衡”导致的内存墙。Level 4 的神话飞跃，正是完美解决了三层总线的带宽卸载：

| **优化级别**           | **核心神技**                 | **打破了什么墙？**                              | **为什么能起飞？**                                           |
| ---------------------- | ---------------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| **Level 1 (Naive)**    | 无                           | 撞死在 **Global Memory** 带宽墙上               | 没有任何复用，算一次去极慢的显存拉一次，总线瘫痪，CUDA Core 全在空转干等。 |
| **Level 2 (Shared)**   | 片上公共水缸 (Shared Memory) | 打破 Global 墙，但撞上 **Shared Memory** 带宽墙 | 32 倍协作复用，显存不卡了。但每做 1 次乘法都要去 Shared 读 2 次，片上内部数据总线瞬间打满，计算单元依然在等数据。 |
| **Level 3 (Float4)**   | 向量化大卡车 (LDG.128)       | 仅加速了外部搬运，计算带宽依然是瓶颈            | **解答了你的疑问：为什么 V3 毫无提升？** 因为当时瓶颈卡在内部 Shared 读太慢，你把外部进货优化得再快，货也只能堆在仓库里吃不进去。 |
| **Level 4 (Register)** | 私有杯子 + 外积魔法 + Float4 | **彻底打破所有带宽墙**，释放满血算力            | **1. 降维打击：** 寄存器外积把内部 Shared 访存压力暴降 8 倍！ **2. 瓶颈转移与 Float4 觉醒：** 计算解放后，V3 留下的 float4（LDG.128 + 动态引入的 LDS.128）成为了刚需，彻底喂饱了外部和内部的搬运带宽，木桶再无短板！ |



## 五、后续优化简述

在准备 AI Infrastructure 或高性能计算（HPC）岗位的面试时，对 Level 4（Register Tiling）之后的优化演进进行归纳是极其高频且加分的硬核考点。面试官往往不只看你做到了多少 TFLOPS，更看重你**在打破旧瓶颈后，如何敏锐地识别新瓶颈，并利用现代硬件特性（Ampere 架构）做极致重叠（Overlap）的思维。**

以下是为你整理的 **Level 4 之后的算子优化路径与核心面试大纲**，可直接转化为面试话术：

### 🚀 核心优化路径演进全景

在 Level 4 引入 **Register Tiling** 之后，虽然通过纯寄存器外积将访存/计算比暴降至 0.25，彻底解放了 CUDA Core，性能飙升至 **7.2 TFLOPS**。但这也意味着硬件瓶颈发生了转移。之后的优化路径可以概括为以下三大战役：

```
Level 4 (Register Tiling) ── 7.2 TFLOPS
   │
   ├─► 细节指令压榨：LDS.128 + __launch_bounds__ ──► 7.9 TFLOPS (CUDA Core 实际上限)
   │
   ├─► 消除流水线空转：cp.async + Double Buffering (双缓冲) ──► 7.5 TFLOPS
   │
   └─► 跨越架构鸿沟：WMMA TF32 (Tensor Core) ──► 7.7 TFLOPS
```

### 📌 路径一：细节指令压榨（LDS.128 +LB）

#### 1. 优化动机与方向

在 Level 4 循环内部，虽然算力爆发，但微观上仍有两点可以压榨：

- **消除冗余读指令**：从 Shared Memory 提水到私有寄存器 `reg_b` 时，原本需要 8 次标量读取（`LDS.32`）。我们通过指针强转，将其改写为 2 次 **`float4` 向量化读取（`LDS.128`）**。
- **锁死硬件资源分配**：引入 `__launch_bounds__(256, 2)` 编译器指示，强行限制 nvcc 编译器的寄存器分配策略，确保一个 SM 内能同时驻留至少 2 个 Block，防止寄存器溢出（Spill）并提升入驻率（Occupancy）。

#### 2. 核心结果

- **性能表现**：4096² 规模下，算力由 7.19 TFLOPS 提升至 **7.86 TFLOPS**（达到了 RTX 3060 Ti FP32 理论峰值的 **49%**）。
- **NCU 洞察**：消除了内层循环中约 78 万条指令发射开销，**IPC 达到了 1.97**（极其接近 Ampere 架构单发射路 1 FP + 1 LD/cycle 的物理极限）。**这代表了手写纯 CUDA Core 算子的性能天花板。**

### 📌 路径二：消除流水线空转（cp.async + 双缓冲）

#### 1. 优化动机与方向

- **打破串行墙**：在之前的版本中，每个 Tile 的执行流是严格串行的：`[加载数据] -> [__syncthreads] -> [外积计算] -> [__syncthreads]`。这导致加载时计算单元空转，计算时总线空转。
- **硬件异步拷贝**：利用 Ampere 架构（sm_80+）引入的 **`cp.async` 硬件异步拷贝指令**，配合双缓冲（Ping-Pong Shared Memory），让线程在干活（计算当前 Tile）的同时，后台的专用 Lsu/Tlc 单元在默默进货（异步加载下一个 Tile），实现**加载与计算的完全重叠**。

##### 2. 核心结果

- **性能表现**：将 K 轴步长泛化至 `BK=16`，大矩阵下稳定跑出 **7.5 TFLOPS**。在 512² 等中等矩阵上，由于延迟隐藏效果最敏感，加速比高达 **+28%**。
- **关键抉择（BK 扫参）**：
  - `BK=8` 时计算窗口太短（仅 512 FMA），无法掩盖显存 700+ 周期延迟，且异步 commit/wait 开销反噬，性能反而下降。
  - `BK=32` 时双缓冲所需的 Shared Memory 飙升至 **64 KB/Block**，直接触发了消费级显卡（100KB SMEM limit）的元数据超限，导致 Launch 失败。
  - **结论**：`BK=16`（32 KB SMEM）是 Ampere 架构上计算隐藏延迟与资源预算相互妥协的完美甜蜜点（Sweet Spot）。

### 📌 路径三：跨越架构鸿沟（WMMA Tensor Core）

#### 1. 优化动机与方向

- **算力升维**：CUDA Core 的 `FFMA` 指令单次只能做 1 次乘加。为了释放 Ampere 架构真正的潜能，我们引入 **Tensor Core 专用的 MMA 阵列**，使用 `nvcuda::wmma` API 在 **TF32 精度模式**下进行 16×16×8 的矩阵乘加。其理论吞吐直接翻倍（16.2T ──► ~32T TFLOPS）。
- **B 矩阵片上转置**：由于 WMMA 的 `matrix_b` 必须是连续的 `col_major` 布局，我们在协同加载阶段将矩阵 B 巧妙地转置存储在共享内存中（`BsT[N][K]`），实现零运行时开销的硬件对齐。

#### 2. 核心结果

- **性能表现**：跑出了 **7.61 TFLOPS**（WMMA 朴素版）和 **7.74 TFLOPS**（WMMA + cp.async 融合版）。
- **NCU 恐怖洞察（为什么没有出现预期的暴涨？）**：
  1. **算力被同步吞噬**：NCU 报告显示 **Warp 空闲率（No Eligible Warp）高达 91.9%**，SM Busy 暴跌至 19%。因为 Tensor Core 算力太强，瞬间就算完了，导致线程有 90% 以上的时间都在 `__syncthreads()` 物理大闸门前空转干等。
  2. **更深层的瓶颈**：瓶颈由全局内存彻底转移到了 **Shared Memory 内部的总线带宽牆** 以及 **同步栅栏开销** 上。

### 💡 面试突围：高频核心 Q&A 话术

#### Q1：为什么你的 Float4（Level 3）对性能毫无提升，但到了 Level 4 之后却成了刚需？

> **答**：这是经典的**阿姆达尔定律（Amdahl's Law）\**体现。在 Level 2/3 时，算子的死穴在\**内层循环内部**：每做 1 次 FMA 必须去 Shared Memory 读 2 次标量，瓶颈死死卡在 Shared Memory 到寄存器的带宽上。此时你用 `float4` 优化外部（Global -> Shared）的进货速度，货也只能堆在仓库里，计算单元吃不进去。
>
> 但到了 Level 4 引入 Register Tiling 释放了计算瓶颈后，内部 Shared Memory 读写降低了 8 倍，此时外部的搬运和指令发射数量（LDG.128/LDS.128）才真正变成新的关键木桶短板。所以，**高阶优化必须建立在核心算术强度被拉高（打破核心瓶颈）的前题下才会有感。**

#### Q2：你的最高性能达到了 7.9 TFLOPS，但 cuBLAS 能跑到 11 TFLOPS 左右（约 68% 峰值），这近 30% 的差距到底差在哪里？

> **答**：手写 CUDA C++ 走到 49% 峰值已经是高层语法的极限，cuBLAS 能跨越到 68% 是因为 NVIDIA 工程师在底层多做了四件极致的事情：
>
> 1. **SASS 汇编级指令调度**：cuBLAS 是手写 SASS 机器码，精细控制了指令的双发射（Dual-issue）、彻底规避了寄存器 Bank 冲突，并将 `FFMA` 4周期的延迟用指令交错完全隐藏。这是编译器 `-O3` 无法匹敌的。
> 2. **异步多阶段流水线（Multi-stage Pipeline）**：我们做的是 2-stage 双缓冲，cuBLAS 借助底层的 `cp.async` 实现了 3-stage 甚至 4-stage 的深度流水线，能够更完美地吸收 DRAM 刷新等硬件抖动。
> 3. **XOR Swizzling 变幻**：对 Shared Memory 地址进行异或打乱，彻底消除了大规模分块（如 BK=32）时引发的高路数 Shared Memory Bank Conflict。
> 4. **更高级的同步机制**：cuBLAS 抛弃了低效的 `__syncthreads()` 块内强同步，采用了基于底层 SASS 实现的异步屏障（类似于高级架构中的 `cuda::barrier`），只同步数据依赖的 Warp，消除了全局 Bubble。





## 六、lds 128

怎么做到的，说一下：消除冗余读指令：从 Shared Memory 提水到私有寄存器 reg_b 时，原本需要 8 次标量读取（LDS.32）。我们通过指针强转，将其改写为 2 次 float4 向量化读取（LDS.128）。



在手写 CUDA Kernel 优化中，从 Shared Memory（片上水缸）将数据加载到寄存器（私有水杯）的这一步，是由编译器映射到底层的 `LDS`（Load Shared）机器指令来完成的。

下面为你拆解，这一步是如何在代码层面通过指针强转，在底层实现从 8 次 `LDS.32` 到 2 次 `LDS.128` 质变调优的：

### 1. 原始的标量读取模式（导致 8 次 `LDS.32`）

在未进行指令向量化优化的常规编写中（即 Level 4 之前的普通寄存器分块逻辑），如果我们在内层循环需要加载 8 个连续的矩阵 B 的元素到 `reg_b` 寄存器里，代码通常会写成一个显式的 `for` 循环，或者手写 8 行标量赋值：

C++

```
// 伪代码：普通的标量加载方式
for (int j = 0; j < 8; ++j) {
    reg_b[j] = Bs[k][thread_c + j]; // 每次循环读取一个 32-bit 的 float
}
```

- **硬件底层的执行流**：由于 `Bs` 是一个 `float` 数组，硬件的指令发射器（Instruction Fetcher）看到这种标量赋值时，会老老实实地连续发射 **8 条 `LDS.32`**（Load Shared 32-bit）物理指令。
- **瓶颈**：1024 或 256 个线程都在并发发射高密集的标量读指令，指令发射总线和发射器的排队槽（Issue Slot）瞬间就被塞满了，从而引发流水线延迟。

### 2. 向量化指针强转模式（实现 2 次 `LDS.128`）

为了打破指令发射的瓶颈，你在 Level 4 算子（`gemm_kernels.cu`）的内层循环里写出了一段极其硬核的、教科书级别的优化代码：

C++

```
// 截取自你的源码：Kernel 4 Register Tiling 内部
float4 vb0 = *reinterpret_cast<float4*>(&Bs[k * BN + (thread_c + 0)]);
float4 vb1 = *reinterpret_cast<float4*>(&Bs[k * BN + (thread_c + 4)]);

reg_b[0] = vb0.x; reg_b[1] = vb0.y; reg_b[2] = vb0.z; reg_b[3] = vb0.w;
reg_b[4] = vb1.x; reg_b[5] = vb1.y; reg_b[6] = vb1.z; reg_b[7] = vb1.w;
```

#### 💡 它是怎么在底层运作的？

1. **取地址**：`&Bs[k * BN + (thread_c + 0)]` 首先拿到了当前线程在 Shared Memory 中需要读取的矩阵 B 连续 8 个元素中**前 4 个元素的首地址**（指向一个标准的 `float`，32 bits）。
2. **指针重新解释（Reinterpret Cast）**：通过 `reinterpret_cast<float4*>`，我们骗过了编译器。我们强行告诉编译器：“别把这个地址当成普通的 `float*`，请把它当成一个 **`float4\*` 向量类型指针**（128 bits，包含 4 个 float）！”
3. **解引用一网打尽**：前面的 `*` 算子对这个 `float4*` 指针进行解引用。此时，编译器和底层硬件就会触发 **`LDS.128`**（Load Shared 128-bit）单条指令。

一条 `LDS.128` 会直接驱动一个 128 位的向量数据总线，**一口气把连续的 16 个字节（4个 float）从共享内存中直接打进线程私有的连续寄存器组（`vb0`）中**。

同理，由于你要加载 8 个元素，第二行代码用同样的方式将后 4 个元素的首地址强转，发射了第二条 `LDS.128`，拿到了 `vb1`。最后，再通过寄存器间的逻辑映射（`.x`, `.y`, `.z`, `.w`）整整齐齐地平铺给 `reg_b`。

### 3. 面试官最爱的硬核拷问：这里有两个天坑，你是怎么避开的？

如果你在面试中只背出了“指针强转实现 `float4`”，面试官可能会追问你两个关于 `float4` 的硬件底层约束。如果你能主动说出这两点，面试直接保送：

#### 坑一：16字节内存对齐（Alignment）约束

- **硬件铁律**：在 CUDA 架构中，任何向量化读写指令（无论是 Global Memory 的 `LDG.128` 还是 Shared Memory 的 `LDS.128`）都必须满足**地址是 16 字节（128-bit）物理对齐的**。如果首地址不能被 16 整除，程序会在运行时直接报 `misaligned address` 错误，面临崩溃。
- **你是怎么做到的**：你的代码中 `#define BN 128`，由于 `BN` 是 128 的倍数（天然对齐 4 的倍数），并且你的每个线程分配的起始列 `thread_c = threadIdx.x * TN`，其中 `TN = 8`。无论是 `k * BN` 还是 `thread_c`，它们作为索引时，对应的 float 数量全部是 4 的倍数（1个 float=4 字节，4个 float=16 字节），这就**天然且完美地保证了每一次强转的起始物理地址都精确踩在了 16 字节对齐线上**。

#### 坑二：指令数量减少了，那寄存器会 Spill（溢出）吗？

- **硬件内部机制**：普通的 `float reg_b[8]` 在编译器的寄存器分配器（Register Allocator）中，不一定会被分配到物理上连续的寄存器里。但当你声明了 `float4 vb0` 时，编译器**必须在硬件上为其分配一块物理上连续的、容量为 4 个 32-bit 的寄存器页**。
- **你是怎么做到的**：你配合使用了 `__launch_bounds__(256, 2)` 指示，强行锁死了每个 Block 只有 256 个线程，并让编译器提前做好寄存器预算规划。这样在做 `float4` 连续物理寄存器分配时，既极大地减少了指令发射器的压力（512次循环 × 256 Tiles 整体砍掉了近 78 万条标量读指令开销），又成功压制住了寄存器溢出风险。这就是让 IPC 直逼 1.97 物理极限的根本原因！



## 七、IPC相关

在准备 CUDA 算子优化相关的面试时，**IPC（Instructions Per Cycle，每周期执行指令数）** 是面试官用来考核你对 GPU 微架构、流水线瓶颈以及 Nsight Compute (NCU) 性能分析理解深度的核心指标。

结合你的优化笔记，我们可以把 IPC 在 GEMM 优化中的演进和核心拷问拆解为以下几个硬核考点：

### 一、 什么是 IPC？在 GPU 架构中它代表什么？

- **基本定义**：IPC 是指 GPU 在每个时钟周期内，流式多处理器（SM）实际执行/发射的指令数量。

- **硬件上限（以你的 Ampere 架构为例）**：Ampere 架构的每个 SM 内部有 4 个 Warp 调度器（Warp Scheduler）。每个调度器在每个周期可以双发射（Dual-issue）两条不同的指令（例如一条浮点计算指令 + 一条访存/跳转指令）。因此，在理想状态下，单个 SM 的理论 IPC 上限可以达到：

  4 调度器×2 发射路=8.0

- **为什么实际值远低于 8.0？**：在实际的 FP32 GEMM 运算中，由于主要消耗的是 FP32 计算单元和片上访存总线，加上指令之间的依赖（Latency）和各类 Stall（长 scoreboard、短 scoreboard、Warp 同步等），手写 CUDA Core 算子的实际活跃 IPC 能稳定达到 **2.0 左右** 就已经触及了硬件调度的物理极限。

### 二、 你的 GEMM 优化历程中，IPC 是如何剧烈变化的？

从 NCU 实测数据来看，IPC 的每一次跳跃都精准地反映了底层瓶颈的转移：

| 优化阶段                       | 实际活跃 IPC    | 底层微架构行为分析                                           |
| ------------------------------ | --------------- | ------------------------------------------------------------ |
| **Level 1 (Naive)**            | **1.10**        | **纯 Memory-Bound**：Warp 调度器中 72.6% 的时间没有可发射的指令（No Eligible Warp）。线程大部分时间都在挂起等待从 Global Memory 拉取数据（Long Scoreboard Stall），导致指令发射器处于饥饿状态，IPC 极其低效。 |
| **Level 2 (Shared Memory)**    | **0.90**        | **瓶颈转移**：虽然成功把 Global Memory 流量砍掉了 32 倍，但由于内层循环每做 1 次乘加就要发起 2 次 Shared Memory 标量读（`LDS.32`），导致**共享内存的数据读取延迟（短 Scoreboard Stall）和 Bank 冲突开始主导流水线**。由于 `LDS.32` 的硬件延迟高于标量 L1 缓存命中，IPC 反而微跌至 0.90。 |
| **Level 4 (Register Tiling)**  | **1.88**        | **爆发性飞跃（ILP 战胜 TLP）**：通过让单个线程接管 8×8 的结果分块，在纯寄存器内部一口气做 64 次外积乘加（FMA）。Shared Memory 的读取频次暴降 8 倍，Warp 调度器的就绪队列被瞬间填满。**此时计算管线和访存管线完美交错，IPC 暴涨至 1.88**。 |
| **Level 5 (LDS.128 +LB 压榨)** | **1.97**        | **触及 CUDA Core 极限**：在 Register Tiling 的基础上，利用指针强转，将内层循环中对矩阵 B 的 8 次 `LDS.32` 标量读取改写为 2 次 `LDS.128`（`float4`）向量化读取。这直接**消除了内层循环中大量的访存指令发射，彻底释放了指令发射器的槽位（Issue Slot）**。IPC 达到了 **1.97**，几乎完美吃满了 Ampere 架构单周期“1条计算 + 1条片上读写”的双发射物理极限。 |
| **Level 6 (WMMA Tensor Core)** | **0.32 / 0.40** | **跨越式指标假象**：引入 Tensor Core 后，IPC 暴跌至 0.32-0.40。**注意：这绝不意味着性能变差了！** 因为一条 Tensor Core 的 `mma_sync` 属于矩阵级大指令，单条指令在底层一周期就能吞掉 16×16×8 次乘加运算。计算单元太快了，导致 90% 以上的时间 Warp 都在 `__syncthreads()` 栅栏前挂起干等（Wait Stall），由于指令总数变少且同步空转严重，IPC 指标自然暴跌。 |

### 三、 面试突围：高频 IPC 核心拷问与话术

### Q1：在 NCU 中，你发现 RegTile 版本的 Occupancy（入驻率）比 Naive 版本暴跌了近一倍（66% ──► 27%），为什么它的 IPC 反而能逆势暴涨、整体算力飚了 7 倍？

> **答**：这正是 CUDA 优化中经典的 **ILP（指令级并行）与 TLP（线程级并行）的权衡取舍**。
>
> - **Naive 版本**是用高 Occupancy（多兵力）来盲目隐藏 Global Memory 的超长延迟（TLP），但人再多，没有数据也只能在流水线里空转，所以活跃 IPC 极低。
> - **RegTile 版本**由于单线程吞了 8×8 的工作量，声明了 64 个累加寄存器，导致每个 Block 消耗的寄存器资源暴涨，在硬件限制下，单 SM 驻留的 Block 数量减少，Occupancy 骤降。但是，因为每个线程手里有充足的数据和高密集的独立寄存器计算逻辑，它的**指令级并行（ILP）极高，单个 Warp Scheduler 根本不需要切换，就能连续发射大量的计算指令喂饱流水线**。
> - **结论**：对于 Compute-Bound 算子，**低 Occupancy + 高 ILP 带来的高 IPC 吞吐**，远比高 Occupancy 但天天卡死挂起的状态更为高效。

### Q2：既然你把 IPC 压榨到了 1.97，几乎到了手写纯 CUDA C++ 的极限，那你怎么看待这个指标的“天花板”？

> **答**：IPC 达到 1.97 说明在**标量 FMA 流水线**上，指令发射器的潜力已经被榨干了（双发射效率接近 100%）。如果想继续突破算力天花板，执着于提高标量 IPC 已经没有意义了，必须改变**指令的架构维度**。
>
> 也就是说，需要从传统 CUDA Core 的标量指令（FMA），切换到 Tensor Core 的矩阵级宏指令（WMMA / MMA）。在 Tensor Core 下，虽然 NCU 统计到的标量 IPC 会因为同步开销和指令集重构而大幅下降（降到 0.4 左右），但它单条指令覆盖的 FLOP 运算量发生了数量级的跃升，这才是跨越内存墙、追求更高真实算力（TFLOPS）的正确路径。
