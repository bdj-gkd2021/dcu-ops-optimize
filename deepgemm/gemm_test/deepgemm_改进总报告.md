# DeepGEMM 算子改进总报告

> 涵盖三大算子修复：contiguous、标准 masked、masked_ll（low-latency）

---

# 第一部分：对DeepGEMM一览

## 1. 改进概览

针对gfx936对 DeepGEMM 源码包完成了三项关键改进，使 MoE 推理链路在正确性和性能上达到可用水平：

| 改进项 | 目标算子 | 问题 | 改进方式 | 受益场景 |
|--------|----------|------|----------|----------|
| **A. contiguous 性能优化** | `m_grouped_w8a8_gemm_nt_contiguous` | 每次调用均执行 `hipHostRegister`（~1900μs），EP 场景下 overhead 反超计算 | mapped host memory pool 持久复用 | prefill / EP 切分场景 |
| **B. 标准 masked 精度修复** | `m_grouped_w8a8_gemm_nt_masked` | ASM kernel grid 硬编码导致精度崩溃（cosine ~0.01），Python wrapper 无视用户 config | 动态 grid 计算 + 尊重 config | sglang 全链路 W8A8 masked 推理 |
| **C. masked_ll 精度修复 + 维度扩展** | `m_grouped_w8a8_gemm_nt_masked_ll` | gfx936 MMA 指令缺失、B operand 重排错误、原 allowlist 不支持 MiniMax-M2.5 TP=8 维度 | MMA 路径补全 + Python pack 阶段子块重排 + N/K/E/CU allowlist 扩展 | decode 小 batch 场景 |

## 2. 关键指标对比

### 2.1 Contiguous：EP 场景 overhead 消除

| E (等效 EP) | 原始版 per-call | 改进后 per-call | 改善 |
|-------------|----------------|----------------|------|
| 256 (EPx1) | 2941 μs | 2893 μs | -1.6% |
| **64 (EPx4)** | 1717 μs | 772 μs | **-55%** |
| **32 (EPx8)** | 1133 μs | 420 μs | **-63%** |
| **16 (EPx16)** | 1188 μs | 256 μs | **-78%** |
| **8 (EPx32)** | 2025 μs | 155 μs | **-92%** |

**端到端 SGLang (TP8 EP8, 51200 input)**：

| 并发 | 原始版吞吐 | 改进后吞吐 | 改善 |
|------|-----------|-----------|------|
| 1 | 52322 tok/s | 66498 tok/s | +27% |
| 16 | 39693 | 47190 | +19% |
| 32 | 33438 | 46597 | **+39%** |
| 64 | 19622 | 45649 | **+133%（2.3x）** |

### 2.2 标准 masked：精度从不可用到完全正确

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| cosine similarity（E=256, M=8/128/1024） | ~0.01 | **1.000000** |
| mode 1000 latency (M=1024, E=256) | 错误输出 | 9.11 ms |
| mode 1000 vs 1002 加速比 | — | **1.7-2.0x** |

### 2.3 masked_ll：gfx936 精度修复 + TP=8 维度支持

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| cosine similarity | 全零输出（MMA 未发出） | **1.000000** |
| 支持维度（N） | {3072, 4096, 6144, 7168} | 新增 **384** |
| 支持维度（K） | {1536, 2048, 3072, 6144, 7168} | 新增 **192** |
| decode perf（max_m≤21, E=32, K=N=3072） | — | vs 标准 masked **1.01-1.41x** |
| K=192 down GEMM（EP=8+TP=8） | 不支持 | B≤256 区间 **1.00-1.86x** |

## 3. 影响范围

| 层级 | 改动项 |
|------|--------|
| **C++ kernel** | 4 个文件：contiguous pool、masked ASM grid、LL MMA + STORE_PERMUTE + native checks + 维度 dispatch |
| **Python wrapper** | 2 个文件：masked config 逻辑、LL weight pack B 子块重排 + 测试脚本维度扩展 |
| **sglang 集成** | 2 个文件（`ep_moe/layer.py`、`compressed_tensors_moe_marlin.py`）：LL decode forward + LL 权重选择性双持 + 环境变量控制开关 |
| **GPU 显存** | contiguous pool：~100KB mapped host memory（可忽略）；LL selective 双持：后半层额外 LL 6-D 权重（显存代价取决于层数和维度） |

---

# 第二部分：技术实现细节

## 一、Contiguous kernel `hipHostRegister` 性能优化

### 1.1 问题

文件：[csrc/py_itfs_cu/asm_m_grouped_w8a8_gemm_nt_contiguous.cu](csrc/py_itfs_cu/asm_m_grouped_w8a8_gemm_nt_contiguous.cu)

每次调用 `_m_grouped_w8a8_gemm_nt_contig_asm_impl` 都执行：

```cpp
GemmInfo<OutputType> *d_prob = malloc(sizeof(GemmInfo<OutputType>));  // 堆分配
hipHostRegister(d_prob, sizeof(...), hipHostRegisterMapped);           // OS 页锁定 + GPU 页表映射
hipHostGetDevicePointer(&d_argsPtr, d_prob, 0);                       // 获取设备端地址
// ... launch kernel ...
// 无 free，无 unregister  →  内存泄漏 + pinned 页泄漏
```

`hipHostRegister` 在 DTK 2604 上约 1900μs/次。在 EP 场景（per-rank E 很小）下，计算量极小的 kernel 调用被 per-call overhead 主导：E=8 时 overhead 占比 >90%。

### 1.2 修复方案：mapped host memory pool

**设计思路**：

- 预分配 1024 个 mapped host memory slot（`hipHostMalloc` + `hipHostMallocMapped`），进程生命周期内复用
- 每次调用从 pool 中 pop 一个 slot，直接写入 mapped host memory（GPU 通过 device pointer 零拷贝可见）
- kernel 完成后通过 `hipStreamAddCallback` 自动归还 slot
- `std::call_once` 保证 pool 初始化只执行一次
- `std::mutex` + `std::condition_variable` 保证多线程/多 stream 安全

**架构对比**：

| | 原始版 | 改进版（pool） |
|---|---|---|
| 参数内存 | per-call `malloc` + `hipHostRegister` | 预分配 1024 mapped host slots |
| GPU 如何读取 | 每次注册映射（~1900μs） | 首次注册后零拷贝读 |
| per-call 分配 | ~1900 μs | pool pop（~ns） |
| slot 归还 | 从不归还（泄漏） | stream callback 自动归还 |
| 多 stream 安全 | ✓（每调独立，但泄漏） | ✓（独立 slot + callback 保证生命周期） |

### 1.3 修复 diff

**新增数据结构（文件头部）：**

```diff
--- a/csrc/py_itfs_cu/asm_m_grouped_w8a8_gemm_nt_contiguous.cu
+++ b/csrc/py_itfs_cu/asm_m_grouped_w8a8_gemm_nt_contiguous.cu
@@ -7,7 +7,9 @@
 #include <vector>
+#include <mutex>
+#include <queue>

+struct __attribute__((packed)) GemmInfoHost {
+    uint32_t m, n, batch, k;
+    void* d;
+    void* c;
+    int8_t* a;
+    int8_t* b;
+    uint32_t strideD1, strideD2, strideC1, strideC2;
+    uint32_t strideA1, strideA2, strideB1, strideB2;
+    int8_t  alpha[16];
+    int8_t   beta[16];
+    float* scaleA;
+    float* scaleB;
+};
```

**Pool 管理 + stream callback：**

```diff
 namespace deepgemm {

+constexpr int POOL_SIZE = 1024;
+
+struct PoolEntry {
+    GemmInfoHost* h_ptr;   // host mapped pointer
+    void* d_ptr;           // device-visible pointer
+};
+
+static std::vector<PoolEntry> g_pool;
+static std::mutex g_pool_mutex;
+static std::condition_variable g_pool_cv;
+static std::once_flag g_pool_init_flag;
+
+static hipStream_t g_callback_stream;
+static hipEvent_t g_callback_event;
+static std::once_flag g_callback_stream_init_flag;
+
+static void init_callback_stream() {
+    std::call_once(g_callback_stream_init_flag, [&]() {
+        hipStreamCreate(&g_callback_stream);
+        hipEventCreate(&g_callback_event);
+    });
+}
+
+static void init_pool() {
+    std::call_once(g_pool_init_flag, [&]() {
+        g_pool.reserve(POOL_SIZE);
+        for (int i = 0; i < POOL_SIZE; i++) {
+            PoolEntry entry;
+            hipHostMalloc(&entry.h_ptr, sizeof(GemmInfoHost),
+                          hipHostMallocMapped);
+            hipHostGetDevicePointer(&entry.d_ptr, entry.h_ptr, 0);
+            g_pool.push_back(entry);
+        }
+    });
+}
+
+inline PoolEntry acquire_from_pool() {
+    init_pool();
+    std::unique_lock<std::mutex> lock(g_pool_mutex);
+    g_pool_cv.wait(lock, [&]() { return !g_pool.empty(); });
+    auto entry = g_pool.back();
+    g_pool.pop_back();
+    return entry;
+}
+
+inline void release_to_pool(PoolEntry entry) {
+    std::lock_guard<std::mutex> lock(g_pool_mutex);
+    g_pool.push_back(entry);
+    g_pool_cv.notify_one();
+}
```

**`_contig_asm_impl` 核心改动：**

```diff
 void _m_grouped_w8a8_gemm_nt_contig_asm_impl(...) {

-    GemmInfo<OutputType> *d_prob;
-    d_prob = (GemmInfo<OutputType> *)malloc(sizeof(GemmInfo<OutputType>));
+    auto entry = acquire_from_pool();

     const int size_m = input.size(0);
     ...

-    {
-    d_prob->m = size_n;
-    d_prob->n = size_m;
-    ...
-    d_prob->a = b_qweight.data_ptr<int8_t>();
-    d_prob->b = input.data_ptr<int8_t>();
-    d_prob->d = static_cast<OutputType*>(output.data_ptr());
-    d_prob->scaleA = b_scale.data_ptr<float>();
-    d_prob->scaleB = a_scale.data_ptr<float>();
-    }
+    entry.h_ptr->m = size_n;
+    entry.h_ptr->n = size_m;
+    ...
+    entry.h_ptr->a = b_qweight.data_ptr<int8_t>();
+    entry.h_ptr->b = input.data_ptr<int8_t>();
+    entry.h_ptr->d = output.data_ptr();
+    entry.h_ptr->scaleA = b_scale.data_ptr<float>();
+    entry.h_ptr->scaleB = a_scale.data_ptr<float>();

+    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
+    void* d_argsPtr = entry.d_ptr;

     ...

-    hipHostRegister(d_prob, sizeof(GemmInfo<OutputType>), hipHostRegisterMapped);
-    uint8_t* d_argsPtr;
-    hipHostGetDevicePointer((void**)&d_argsPtr, d_prob, 0);

     hipFunctionArgs.gemm_count = 1;
-    hipFunctionArgs.DeviceUserArguments = d_argsPtr;
+    hipFunctionArgs.DeviceUserArguments = (void const*)d_argsPtr;

     ...
-    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
-    char func_name_fp[1024], fp16_co_file_tn[1024], ...;   // 每次 memset+sprintf
-    memset(func_name_fp, 0x0, sizeof(func_name_fp));
-    ...
-    sprintf(func_name_bf, "DeepGemm_W8A8..._BF16", BLOCKM, BLOCKN, BLOCKK);
-    ...
+    static char func_name_fp[1024], ...;                    // 仅首次初始化
+    static std::once_flag init_flag;
+    std::call_once(init_flag, [&]() {
+        memset(func_name_fp, 0x0, sizeof(func_name_fp));
+        ...
+        sprintf(func_name_bf, "DeepGemm_W8A8..._BF16", BLOCKM, BLOCKN, BLOCKK);
+        ...
+    });

     ...
     impl_ptr->launch_kernel_ext({ ... });

+    // --- stream callback: kernel 完成后归还 slot ---
+    init_callback_stream();
+    hipEventRecord(g_callback_event, stream);
+    hipStreamWaitEvent(g_callback_stream, g_callback_event, 0);
+
+    PoolEntry* e = new PoolEntry(entry);
+    hipStreamAddCallback(g_callback_stream,
+        [](hipStream_t, hipError_t, void* userData) {
+            PoolEntry* e = static_cast<PoolEntry*>(userData);
+            release_to_pool(*e);
+            delete e;
+        }, e, 0);
+
     return;
 }
```

### 1.4 修复后效果

**算子微基准（`bench_decode_overhead.py`）：**

| E | 等效 EP | 原始版 | 改进后 | 改善 |
|---:|---------|--------:|--------:|------:|
| 256 | EPx1 | 2941 μs | 2893 μs | -1.6% |
| 64 | EPx4 | 1717 μs | 772 μs | **-55%** |
| 32 | EPx8 | 1133 μs | 420 μs | **-63%** |
| 16 | EPx16 | 1188 μs | 256 μs | **-78%** |
| 8 | EPx32 | 2025 μs | 155 μs | **-92%** |

Pure overhead（2000 次相同调用）：1937 μs → 153 μs（**12.6x**）

**端到端 SGLang 吞吐（TP8 EP8, 75% prefix cache, 51200 input）：**

| 并发 | 原始版 | 改进后 | 改善 |
|------|--------:|--------:|------:|
| 1 | 52322 tok/s | 66498 tok/s | +27% |
| 16 | 39693 | 47190 | +19% |
| 32 | 33438 | 46597 | **+39%** |
| 64 | 19622 | 45649 | **+133%（2.3x）** |

扩展性（并发 1→64 吞吐衰减）：原始版 -63% → 改进后 **-31%**

**压力测试（race 检测）：**

| 测试模式 | 错误数 |
|---------|--------|
| multi_stream（双 stream） | **0/2000** |
| rapid_fire（单 stream 交替 5000 次） | **0/5000** |
| multi_thread（4 线程） | **0/2000** |

## 二、标准 masked kernel 精度修复

### 2.1 问题

`m_grouped_w8a8_gemm_nt_masked` 在 gfx936上精度崩溃，cosine similarity 约 0.01（接近随机输出）。根因有两个：

**Bug 1 — Python wrapper 忽略用户 config**

文件：[deepgemm/m_group_gemm_nt_masked.py](deepgemm/m_group_gemm_nt_masked.py)

- sglang 和测试代码显式传 `config={'MODE': 1000}` 希望使用 ASM 256×256×128 kernel
- 旧代码无视用户传入的 config，始终重新调用 `deepgemm_masked_config_i8fp8(...)` 自动选 mode
- 大 M 场景 fallback 到 mode 1002（256×64×128），性能比 mode 1000 慢约 2x

**Bug 2 — ASM kernel grid 硬编码**

文件：[csrc/py_itfs_cu/m_grouped_w8a8_gemm_nt_masked.cu](csrc/py_itfs_cu/m_grouped_w8a8_gemm_nt_masked.cu)

- grid 的 `gdx=128, gdy=1, gdz=1` 被硬编码
- ASM kernel 用 `gdy` 判断 N 方向 tile 数（`problemNumGroupTiles1 = gdy`）
- N=3072、mode 1000（BLOCKM=256）时实际有 12 个 N-tile，但 `gdy=1` 让 kernel 只处理了 1 个，剩下 11/12 未计算

### 2.2 修复 diff

**修复 1 — Python wrapper 尊重 config：**

```diff
--- a/deepgemm/m_group_gemm_nt_masked.py
+++ b/deepgemm/m_group_gemm_nt_masked.py
@@ -148,8 +148,11 @@ def m_grouped_w8a8_gemm_nt_masked(...):
-    size_k = a[0].shape[2]
-    mode = deepgemm_masked_config_i8fp8(expected_m_per_group, size_k)
+    if config is not None and 'MODE' in config:
+        mode = config['MODE']
+    else:
+        size_k = a[0].shape[2]
+        mode = deepgemm_masked_config_i8fp8(expected_m_per_group, size_k)
```

**修复 2 — ASM kernel grid 动态计算：**

```diff
--- a/csrc/py_itfs_cu/m_grouped_w8a8_gemm_nt_masked.cu
+++ b/csrc/py_itfs_cu/m_grouped_w8a8_gemm_nt_masked.cu
@@ -76,10 +76,9 @@ void _m_grouped_marlin_w8a8_gemm_nt_masked_asm_impl(...) {
     /* grid sizes */
     size_t globalWorkSize[3] = {1, 1, 1};
-    // globalWorkSize[0] = DIVIDE(need_size_m, BLOCKN);
-    // globalWorkSize[1] = DIVIDE(size_n, BLOCKM);
-    // globalWorkSize[2] = experts_num;
-
-    globalWorkSize[0] = 128;
+    globalWorkSize[0] = DIVIDE(need_size_m, BLOCKN);
+    globalWorkSize[1] = DIVIDE(size_n, BLOCKM);
+    globalWorkSize[2] = experts_num;
 }
```

### 2.3 修复后效果

精度全部恢复（DCU-04/gfx936）：

```
E=256: M=8/128/1024  → cosine=1.000000, PASS
E=32:  M=8/128/1024  → cosine=1.000000, PASS
```

mode 1000 vs mode 1002 性能对比（E=256）：

| M | mode 1000 | mode 1002 | 加速比 |
|---:|----------:|----------:|-------:|
| 8 | 2.30 ms | 4.08 ms | **1.77x** |
| 128 | 2.65 ms | 4.62 ms | **1.74x** |
| 1024 | 9.11 ms | 18.72 ms | **2.05x** |
| 4096 | 37.70 ms | 76.31 ms | **2.02x** |

---

## 三、masked_ll (low-latency) kernel 修复与扩展

### 3.1 问题

`m_grouped_w8a8_gemm_nt_masked_ll` 是面向 decode 小 batch 的 low-latency grouped GEMM 路径，使用 6-D W6 weight layout。在 gfx936 上存在多个问题：

| # | 问题 | 影响 |
|---|------|------|
| 1 | **gfx936 MMA 指令缺失** — 只有 `#if defined(__gfx938__)` 路径，gfx936 上 MMA 指令从未发出 | 累加器保持零值，输出全零 |
| 2 | **B operand N 子块顺序不匹配** — gfx936 16-lane 需要 `0,4,8,12,1,5,...` 顺序，W6 layout 是连续 `0,1,2,...` | cosine ~0.25，B 读取错位 |
| 3 | **native 入口缺检查** — 无 contiguous/dtype/dim 校验 | 非 contiguous tensor 静默错误 |
| 4 | **allowlist 不含 TP=8 维度** — MiniMax-M2.5 EP=8+TP=8 的 `N=384, K=192` 不在支持列表 | launch 失败 |

### 3.2 修复 1：gfx936 MMA 指令补全

文件：[csrc/include/low_latency_fp8_masked_utils.h](csrc/include/low_latency_fp8_masked_utils.h)

```diff
--- a/csrc/include/low_latency_fp8_masked_utils.h
+++ b/csrc/include/low_latency_fp8_masked_utils.h
@@ -23,8 +23,11 @@ __device__ __forceinline__ void mmac_(...) {
 #if defined(__gfx938__)
     v3 = __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts(
         v1, v2, v3, 1, 0, 0);
+#elif defined(__gfx936__) || defined(__gfx928__)
+    v3 = __builtin_hcu_mmac_i32_16x16x32_i8(v1, v2, v3);
 #endif
 }
```

### 3.3 修复 2：B operand N 子块重排

**根因**：gfx936 的 16-lane B load 需要 N 子块按 `0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15` 排列，而原始 W6 layout 的 `n_in` 轴是连续 `0,1,2,...,15`。

**方案**：在 Python pack 阶段做一次性权重重排，避免每次 kernel 热路径做地址变换。

**Python pack 侧** — 文件：[deepgemm/m_group_gemm.py](deepgemm/m_group_gemm.py)

```diff
--- a/deepgemm/m_group_gemm.py
+++ b/deepgemm/m_group_gemm.py
@@ -220,6 +220,22 @@ def _logical_int8_weight_2d_to_w6(...):
     w6 = weight_enk.reshape(...).permute(...)
+    n_perm = torch.tensor(
+        (
+            0, 4, 8, 12,
+            1, 5, 9, 13,
+            2, 6, 10, 14,
+            3, 7, 11, 15,
+        ),
+        dtype=torch.long,
+        device=w6.device,
+    )
+    return w6.index_select(4, n_perm).contiguous()
-    return w6.contiguous()
```

**C++ kernel 侧** — 文件：[csrc/include/low_latency_fp8_masked.h](csrc/include/low_latency_fp8_masked.h)

kernel 内部的 `STORE_PERMUTE` 在主要 kernel 路径中已设为 `false`（权重已在 pack 阶段排好序）。

### 3.4 修复 3：native 入口检查

文件：[csrc/include/low_latency_fp8_masked.h](csrc/include/low_latency_fp8_masked.h)

```diff
--- a/csrc/include/low_latency_fp8_masked.h
+++ b/csrc/include/low_latency_fp8_masked.h
@@ -860,8 +860,18 @@ void masked_fp8_gemm(...) {
+    TORCH_CHECK(matrix_a.is_contiguous(), "matrix_a must be contiguous");
+    TORCH_CHECK(matrix_b.is_contiguous(), "matrix_b must be contiguous");
+    TORCH_CHECK(matri_a_scale.is_contiguous(), "matri_a_scale must be contiguous");
+    TORCH_CHECK(matrixb_scale.is_contiguous(), "matrixb_scale must be contiguous");
+    TORCH_CHECK(matrix_c.is_contiguous(), "matrix_c must be contiguous");
-    TORCH_CHECK(matri_a_scale.dim() == b_block_wise ? 3 : 2, ...);
-    TORCH_CHECK(matrixb_scale.dim() == b_block_wise ? 3 : 2, ...);
+    TORCH_CHECK(matri_a_scale.dim() == (b_block_wise ? 3 : 2), ...);
+    TORCH_CHECK(matrixb_scale.dim() == (b_block_wise ? 3 : 2), ...);
+    TORCH_CHECK(actual_tokens.scalar_type() == at::kInt, ...);
+    TORCH_CHECK(matrix_c.scalar_type() == at::kBFloat16, ...);
 }
```

关键修正：
- 补齐所有 tensor 的 `is_contiguous()` 检查（LL kernel 内部用裸指针 + 线性 offset，非 contiguous 直接错）
- 修复 `dim() == b_block_wise ? 3 : 2` 运算符优先级 bug（应为 `== (b_block_wise ? 3 : 2)`）
- 新增 `actual_tokens` 必须为 int32、`matrix_c` 必须为 bf16 的检查

### 3.5 修复 4：TP=8 维度扩展

MiniMax-M2.5 EP=8+TP=8 的 per-rank 维度：

| GEMM | local E | K | N | 原始 LL 支持？ |
|------|--------:|---:|---:|---------------|
| gate+up | 32 | 3072 | **384** | ❌ N=384 不在 allowlist |
| down | 32 | **192** | 3072 | ❌ K=192 不在 allowlist |

**修改文件**：

| 文件 | 修改内容 |
|------|----------|
| `low_latency_fp8_masked_utils.h` | 新增 `K=192`、`N=384`、`E=256`、`CU=80` dispatch 分支 |
| `low_latency_fp8_masked.h` | kernel 模板参数：`BLOCK_N=128`、`K_SCALE_RANGE=64`（保证 N=384、K=192 整除） |
| `test_moe_deepgemm_w8a8_ll.py` | 新增多维度多 CU 精度覆盖测试 |

**dispatch 扩展 diff：**

```diff
--- a/csrc/include/low_latency_fp8_masked_utils.h
+++ b/csrc/include/low_latency_fp8_masked_utils.h
@@ -61,6 +61,9 @@
 #define MOE_LL_K_SWITCH(k, ...) \
   [&] { \
+  if (k == 192) { constexpr static int K = 192; return __VA_ARGS__(); } \
+  else if (k == 1536) { constexpr static int K = 1536; return __VA_ARGS__(); } \
   ...
   } }();

@@ -82,6 +85,8 @@
 #define MOE_LL_N_SWITCH(n, ...) \
   [&] { \
+  if (n == 384) { constexpr static int N = 384; return __VA_ARGS__(); } \
+  else if (n == 3072) { constexpr static int N = 3072; return __VA_ARGS__(); } \
   ...
   } }();

@@ -100,6 +105,8 @@
 #define MOE_LL_E_SWITCH(num_expert, ...) \
   [&] { \
   ...
+  else if (num_expert == 256) { constexpr static int EXPERTS = 256; return __VA_ARGS__(); } \
   } }();

@@ -118,6 +125,8 @@
 #define MOE_LL_CU_SWITCH(cu, ...) \
   [&] { \
+  else if (cu == 80) { constexpr static int CUs = 80; return __VA_ARGS__(); } \
   ...
   } }();
```

**测试脚本扩展 diff：**

```diff
--- a/gemm_test/test_moe_deepgemm_w8a8_ll.py
+++ b/gemm_test/test_moe_deepgemm_w8a8_ll.py
@@ -68,7 +68,12 @@ def test_precision():
-    configs = [(3072, 3072)]
-    cu_values = [128]
+    configs = [
+        (3072, 3072, "GEMM1 EP=8 only"),
+        (3072, 384,  "GEMM1 EP=8+TP=8"),
+        (192, 3072,  "GEMM2 EP=8+TP=8"),
+    ]
+    cu_values = [64, 80, 128, 256]
```

### 3.6 修复后效果

**精度（gfx936）：**

```
基础维度 (K=N=3072, E=16/32, M=8/128/1024): cosine=1.000000, 全 PASS
TP=8 GEMM1 (K=3072, N=384): M={8,128,1024}, CU=64/80/128/256, 全 PASS
TP=8 GEMM2 (K=192,  N=3072): M={8,128,1024}, CU=64/80/128/256, 全 PASS
```

**算子性能（E=32, K=N=3072, gfx938 参考）：**

decode 分布模拟，LL vs 标准 masked speedup（>1 表示 LL 更快）：

| B | max_m | speedup | 结论 |
|--:|------:|--------:|------|
| 1 | 1 | **1.39x** | LL |
| 8 | 4 | **1.14x** | LL |
| 32 | 13 | **1.13x** | LL |
| 64 | 21 | 1.02 | 持平 |
| 128 | 45 | 0.72 | masked 反超 |
| 256 | 76 | 0.46 | masked |

**结论**：LL kernel 在 `max_m ≤ 21` 时通常更快，`max_m ≥ 45` 后标准 masked 更稳。K=192 的 down GEMM（TP=8）在 B≤256 区间 LL 优势更大（1.0-1.86x）。

### 3.7 当前 allowlist（扩展后）

| 维度 | 允许值 |
|------|--------|
| E | {1, 16, 32, 256} |
| N | {384, 3072, 4096, 6144, 7168} |
| K | {192, 1536, 2048, 3072, 6144, 7168} |
| CU | {64, 80, 128, 256} |

> **注意**：LL kernel 不是通用 grouped GEMM。不命中 allowlist 时只打印 unsupported，不会自动 fallback 到标准 masked。EP=4 (E=64) 场景目前不支持。

---

## 四、sglang 端到端集成尝试（masked_ll 路径）

> 以下为 masked_ll 在 sglang 中的接入实验，当前不是最终推荐方案。核心挑战是 prefill 和 decode 需要的 weight layout 不同。

### 4.1 修改文件与内容

| 文件 | 修改 |
|------|------|
| `sglang/.../ep_moe/layer.py` | 新增 `forward_groupgemm_w8a8_marlin_masked_ll` decode forward；wrapper 做 3-D scale squeeze→2-D + `masked_m`→int32 转换；环境变量 `SGLANG_USE_W8A8_MARLIN_LL` 控制开关 |
| `sglang/.../compressed_tensors_moe_marlin.py` | `process_weights_after_loading` 中增加 LL 6-D weight 打包逻辑；支持 `SGLANG_LL_START_LAYER` 选择性双持 + `SGLANG_LL_SKIP_MARLIN` 纯 LL 实验模式 |

### 4.2 关键发现

1. **不能全层纯 LL**：prefill contiguous 路径需要 Marlin layout，如果全层只保留 LL 6-D weight，prefill 会拿到错误 layout 的数据，从第一阶段 hidden states 开始污染，后续 decode 输出乱码。

2. **双持 LL 比 baseline 慢 ~5.5-6.4%**：原因可能是后半层同时持有 Marlin + LL 6-D 两套权重，增加显存和带宽压力；算子 micro benchmark 的收益被 wrapper 开销（`squeeze().contiguous()`）和通信/调度稀释。

3. **PD 分离是更合理的方向**：prefill 节点 Marlin only，decode 节点 LL 6-D only，各自只持有一份权重。

### 4.3 环境变量开关

```bash
# 选择性 LL（后半层双持）
export SGLANG_USE_W8A8_MARLIN_LL=1
export SGLANG_LL_START_LAYER=24

# 纯 LL 实验（prefill 会错误，不可用于正确推理）
export SGLANG_USE_W8A8_MARLIN_LL=1
export SGLANG_LL_SKIP_MARLIN=1

# 恢复 baseline
unset SGLANG_USE_W8A8_MARLIN_LL
unset SGLANG_LL_SKIP_MARLIN
unset SGLANG_LL_START_LAYER
```

---

## 五、修改文件清单

| 优先级 | 文件 | 改进项 |
|--------|------|--------|
| ★★★ | `csrc/py_itfs_cu/asm_m_grouped_w8a8_gemm_nt_contiguous.cu` | contiguous pool 机制 |
| ★★★ | `csrc/py_itfs_cu/m_grouped_w8a8_gemm_nt_masked.cu` | masked ASM grid 动态计算 |
| ★★★ | `deepgemm/m_group_gemm_nt_masked.py` | masked Python config 修复 |
| ★★★ | `csrc/include/low_latency_fp8_masked_utils.h` | gfx936 MMA + 维度 dispatch 扩展 |
| ★★★ | `csrc/include/low_latency_fp8_masked.h` | STORE_PERMUTE + native 检查 + 模板参数调整 |
| ★★★ | `deepgemm/m_group_gemm.py` | W6 weight pack B 子块重排 |
| ★★ | `gemm_test/test_moe_deepgemm_w8a8_ll.py` | 多维度多 CU 测试覆盖 |
| ★ | `sglang/.../ep_moe/layer.py` | sglang LL decode forward |
| ★ | `sglang/.../compressed_tensors_moe_marlin.py` | LL 权重打包 + 选择性双持 |

---

# 第三部分：待向海光确认的问题（masked_ll）

在决定要不要把 sglang decode 路径改成 masked_ll 之前，需要先回答以下问题。这些问题需要向海光负责 deepgemm / sglang 适配的同事确认。

## 1. masked_ll 在 gfx938 上的真实加速窗口

**问题**：他们在 gfx938 上观察到 masked_ll 相比标准 masked 的真实加速窗口是多少，是否也只在小 `masked_m` decode 场景成立？

**我们的观测**（gfx938/DCU-03 micro benchmark，`E=32, N=3072, K=3072, CU=128`）：

固定每个 expert 相同 M：

| M | LL (ms) | masked (ms) | speedup | 结论 |
|---:|---:|---:|---:|---|
| 8 | 0.307 | 0.342 | 1.11x | LL 略快 |
| 128 | 1.463 | 0.376 | 0.26x | masked 明显更快 |
| 1024 | 14.052 | 1.217 | 0.09x | masked 明显更快 |
| 4096 | 62.519 | 4.757 | 0.08x | masked 明显更快 |

LL 不是"大 M 高吞吐"kernel，它只适合 decode 阶段每个 expert 分到很少 token 的情况。

decode 分布模拟，EP=8 未做 TP split，`CU=128`：

| GEMM size | E | 成立区间 | 典型 speedup | 失效区间 |
|------|---:|------|------|------|
| `K=3072, N=3072` | 32 | B=1-64，`max_m<=21` | 1.01-1.41x | B>=128，`max_m>=45` |
| `K=3072, N=3072` | 16 | B=1-32 基本成立，B=64 接近持平 | 1.07-1.38x | B>=128 |
| `K=1536, N=3072` | 32 | B=1-128，`max_m<=45` | 1.00-1.46x | B>=256 |
| `K=1536, N=3072` | 16 | B=1-64，`max_m<=40` | 1.12-1.40x | B>=128 |

关键结论：

- `K=1536, N=3072` 的 down GEMM 比 `K=3072, N=3072` 更适合 LL，因为 K 更小，标准 masked 的 Marlin tile 吞吐优势没那么容易发挥。
- 一旦 `max_m` 到 45 以上，LL 的优势开始消失；到 76/147 这类区间时，标准 masked 通常明显更快。

**待确认**：海光在 gfx938 上的实测数据是否与上述一致？他们是否在更大 batch 或不同 `(E,N,K)` 组合下观察到 LL 有更大的加速窗口？

## 2. sglang 中 contiguous 路径和 masked_ll 路径 weight layout 兼容方案

**问题**：他们在 sglang 里如何解决 contiguous 路径和 masked_ll 路径 weight layout 不同的问题？也就是 prefill 需要 Marlin/contiguous 可用权重，而 decode masked_ll 需要 W6 6-D 权重，这两套 layout 是双持、按需转换、PD 分离，还是有统一的权重表示？

**当前已知的约束**：

sglang P+D 合并部署时有两类 MoE 路径：

```
prefill → DeepEPNormalDispatchOutput → contiguous path → 需要 Marlin layout
decode  → DeepEPLLDispatchOutput     → masked path     → 可以使用 LL 6-D layout
```

- 全层纯 LL 6-D 会导致 prefill 走错 layout，hidden states 从第一阶段开始污染。
- 双持两套权重增加显存和带宽压力，实测比 baseline 慢 ~5.5-6.4%。

**待确认**：海光在 sglang 中具体采用什么方案？如果已经接入 masked_ll，需要说明 contiguous/prefill 和 masked_ll/decode 两条源码路径是否共用同一份权重。

## 3. allowlist 不命中时的 fallback 策略

**问题**：当 `(E,N,K)` 不命中 masked_ll allowlist 时，他们是显式 fallback 到标准 masked，还是在模型加载阶段直接禁用 LL？

**当前代码行为**：gfx936 实现是硬编码 dispatch，不命中 allowlist 时只打印 unsupported，不会自动 fallback。

**当前 allowlist**：

| 参数 | 允许值 |
|------|--------|
| E | {1, 16, 32, 256} |
| N | {384, 3072, 4096, 6144, 7168} |
| K | {192, 1536, 2048, 3072, 6144, 7168} |
| CU | {64, 80, 128, 256} |

**MiniMax-M2.5 匹配情况**：

| 场景 | local E | K | N | 支持状态 |
|------|--------:|---:|---:|----------|
| EP=8，gate+up | 32 | 3072 | 3072 | ✅ 支持 |
| EP=8，down | 32 | 1536 | 3072 | ✅ 支持 |
| EP=8+TP=8，gate+up | 32 | 3072 | 384 | ✅ 支持（依赖新增 N=384） |
| EP=8+TP=8，down | 32 | 192 | 3072 | ✅ 支持（依赖新增 K=192） |
| EP=4 | 64 | 3072/1536 | 3072 | ❌ E=64 不在 allowlist |
| 其他 TP 比例 N=768/512/256 或 K=384/256/128 | 视情况 | 多数不在 | 多数不在 | ❌ 需新增模板分支 |

**待确认**：海光是否有自动 fallback 机制？对于 EP=4 或其他不命中 allowlist 的场景，他们的推荐方案是什么（扩展 allowlist 还是自动 fallback）？
