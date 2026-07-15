# m_grouped_w8a8_gemm_nt_masked_ll gfx936 修复记录

日期：2026-06-07

## 背景

目标是把 W8A8 masked grouped GEMM 从普通 masked 路径：

```python
m_grouped_w8a8_gemm_nt_masked
```

切到 low-latency masked 路径：

```python
m_grouped_w8a8_gemm_nt_masked_ll
```

当前验证的主要场景是 `block_wise=False`，输入约定如下：

- A: `[E, M, K]`
- B: 6-D packed weight，来自 `pack_int8_weight_enk_to_w6_low_latency`
- C: `[E, M, N]`
- A scale: `[E, M]`
- B scale: `[E, N]`
- `masked_m`: `[E]`，native 侧按 `int32_t*` 读取

最初 gfx936 上的问题分两层：

- `low_latency_fp8_masked_utils.h` 缺少 gfx936/gfx928 的 HCU int8 MMA 分支时，kernel 没有真正执行 int8 MMA，输出异常。
- 补上 MMA 后，输出不再全 0，但 cosine 稳定在约 0.25，说明 B operand 的 16-lane N 子块读取顺序仍然和 W6 weight layout 不一致。

## 当前最终方案

### 1. gfx936 使用 HCU int8 MMA

文件：

```text
csrc/include/low_latency_fp8_masked_utils.h
```

`mmac_(const vec<int8_t, 8>&, const vec<int8_t, 8>&, vec<int, 4>&)` 需要在 gfx936/gfx928 上调用 HCU int8 MMA builtin：

```cpp
#if defined(__gfx938__)
  v3 = __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts(v1, v2, v3, 1, 0, 0);
#elif defined(__gfx936__) || defined(__gfx928__)
  v3 = __builtin_hcu_mmac_i32_16x16x32_i8(v1, v2, v3);
#endif
```

原因：

- gfx936 不能只落到空分支或不匹配的普通 AMDGCN MMA。
- 同仓库 `csrc/include/moe_marlin/intrinsic_2.h` 中 gfx936 也使用 `__builtin_hcu_mmac_i32_16x16x32_i8`。

### 2. 把 N 子块重排前移到 Python pack 阶段

文件：

```text
deepgemm/m_group_gemm.py
csrc/include/low_latency_fp8_masked.h
```

旧方案是在 kernel 的 B load 地址计算里打开：

```cpp
constexpr bool STORE_PERMUTE = true;
ld_row_N = (ld_row_N / 4) + (ld_row_N % 4) * 4;
```

新方案改成：

```cpp
constexpr bool STORE_PERMUTE = false;
```

同时在 Python pack W6 layout 时提前重排 `n_in` 轴：

```python
_CUSTOMER_N_GFX936_LD_PERMUTE = (
    0, 4, 8, 12,
    1, 5, 9, 13,
    2, 6, 10, 14,
    3, 7, 11, 15,
)

n_perm = torch.tensor(
    _CUSTOMER_N_GFX936_LD_PERMUTE,
    dtype=torch.long,
    device=w6.device,
)
return w6.index_select(4, n_perm).contiguous()
```

对应 W6 layout：

```text
[E, N, K]
  -> reshape [E, n_out, n_in, k_out, k_mid, k_in]
  -> permute [E, k_out, n_out, k_mid, n_in, k_in]
  -> 在 axis=4 的 n_in 上做 gfx936 load 顺序重排
```

原因：

- gfx936 的 B operand load 需要的 `n_in` 顺序是 `0,4,8,12,1,5,...,15`，不是逻辑连续 `0,1,2,...,15`。
- 把这个重排做在 pack 阶段后，kernel 热路径可以按 `ld_row_N = ld_row` 直接读取已经排好的 W6 数据。
- 当前实际调用 `m_grouped_w8a8_gemm_nt_masked_ll(..., block_wise=False)` 走的是普通 `kGroupGemm`，不是 `kGroupGemm_aggressive`。因此只改普通 `kGroupGemm` 的 `STORE_PERMUTE=false` 即可。
- `kGroupGemm_aggressive` 中的 `STORE_PERMUTE=true` 当前没有动，因为它不是这条 non-blockwise masked LL 路径的实际 launch 分支。

如果后续某个分支重新出现 cosine 约 0.25，优先检查 Python pack 的 permutation 方向；如果方向反了，只需要把 `_CUSTOMER_N_GFX936_LD_PERMUTE` 改成 inverse，不需要再改 C++ kernel。

### 3. native 入口参数检查

文件：

```text
csrc/include/low_latency_fp8_masked.h
```

`masked_fp8_gemm` 入口保留 contiguous、维度和 dtype 检查：

```cpp
TORCH_CHECK(matrix_a.is_contiguous(), ...);
TORCH_CHECK(matrix_b.is_contiguous(), ...);
TORCH_CHECK(matri_a_scale.is_contiguous(), ...);
TORCH_CHECK(matrixb_scale.is_contiguous(), ...);
TORCH_CHECK(matrix_c.is_contiguous(), ...);
TORCH_CHECK(matri_a_scale.dim() == (b_block_wise ? 3 : 2), ...);
TORCH_CHECK(matrixb_scale.dim() == (b_block_wise ? 3 : 2), ...);
TORCH_CHECK(actual_tokens.scalar_type() == at::kInt, ...);
TORCH_CHECK(matrix_c.scalar_type() == at::kBFloat16, ...);
```

原因：

- 原来的 `matri_a_scale.dim() == b_block_wise ? 3 : 2` 有 C++ 运算符优先级问题，需要加括号。
- LL kernel 使用裸指针和线性 offset，非 contiguous tensor 会直接造成错误结果。
- `actual_tokens` 在 kernel 里按 `int32_t*` 使用，必须是 int32。
- 当前写出固定为 bf16。

## 编译安装注意

在 DCU 环境同步代码后建议重新rm build后并重装 wheel：

```bash
cd /workspace/deepgemm

rm -rf build 

MAX_JOBS=32 PYTORCH_ROCM_ARCH=gfx936 python3 setup.py bdist_wheel

rm -rf /usr/local/lib/python3.10/dist-packages/deepgemm
rm -rf /usr/local/lib/python3.10/dist-packages/deepgemm-*.dist-info
rm -rf /usr/local/lib/python3.10/dist-packages/deepgemm-*.egg-info

python3 -m pip install --force-reinstall --no-deps dist/*.whl
```

## 精度验证

命令：

```bash
cd /workspace/gemm_test
python test_moe_deepgemm_w8a8_ll.py precision
```

时间：2026-06-07 15:26

| E | M | output shape | cosine | max_abs | max_rel | 状态 |
|---:|---:|---|---:|---:|---:|---|
| 32 | 8 | `[32, 256, 3072]` | 1.000000 | 0 | 0 | PASS |
| 32 | 128 | `[32, 256, 3072]` | 1.000000 | 0 | 0 | PASS |
| 32 | 1024 | `[32, 1024, 3072]` | 1.000000 | 0 | 0 | PASS |
| 16 | 8 | `[16, 256, 3072]` | 1.000000 | 0 | 0 | PASS |
| 16 | 128 | `[16, 256, 3072]` | 1.000000 | 0 | 0 | PASS |
| 16 | 1024 | `[16, 1024, 3072]` | 1.000000 | 0 | 0 | PASS |

结论：新 pack permutation + `STORE_PERMUTE=false` 后，gfx936 上 `m_grouped_w8a8_gemm_nt_masked_ll` 和 PyTorch reference 完全一致。

## 性能对比

下面 `speedup = masked_time / LL_time`：

- `speedup > 1` 表示 LL 更快。
- `speedup < 1` 表示普通 masked 更快。

### 1. load permutation 前移前后对比

测试命令：

```bash
cd /workspace/gemm_test
CU=128 python test_ll_vs_masked.py decode_d
```

测试条件：`decode_d` 模式，**warmup=10, measure=50**（每次计时前 10 次预热，取 50 次测量平均）。所有延迟均为单步 MoE GEMM（不含 attention/all-to-all）。

对比两版已经通过精度验证的 LL 实现：

- 旧版：Python pack 不提前重排 `n_in`，在普通 `kGroupGemm` 的 B load 地址计算里使用 `STORE_PERMUTE=true`。
- 新版：Python pack 阶段提前重排 W6 layout 的 `n_in` 轴，普通 `kGroupGemm` 使用 `STORE_PERMUTE=false`。

下表比较的是 LL 单步延迟；`提升` = `旧版 LL / 新版 LL`。两组数据来自不同时间的 DCU 实测，有少量测量噪声，但趋势明确。

#### E=32, CU=128

| B/conc | max_m | 旧版 LL(ms) | 新版 LL(ms) | 提升 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.115 | 0.087 | 1.32x |
| 2 | 2 | 0.160 | 0.126 | 1.27x |
| 4 | 3 | 0.212 | 0.189 | 1.12x |
| 8 | 4 | 0.322 | 0.284 | 1.13x |
| 16 | 11 | 0.326 | 0.294 | 1.11x |
| 32 | 13 | 0.330 | 0.301 | 1.10x |
| 64 | 21 | 0.404 | 0.342 | 1.18x |
| 128 | 45 | 0.592 | 0.483 | 1.23x |
| 256 | 76 | 1.012 | 0.772 | 1.31x |

#### E=16, CU=128

| B/conc | max_m | 旧版 LL(ms) | 新版 LL(ms) | 提升 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.118 | 0.088 | 1.34x |
| 2 | 2 | 0.154 | 0.120 | 1.28x |
| 4 | 3 | 0.162 | 0.147 | 1.10x |
| 8 | 6 | 0.160 | 0.147 | 1.09x |
| 16 | 10 | 0.160 | 0.147 | 1.09x |
| 32 | 21 | 0.221 | 0.183 | 1.21x |
| 64 | 40 | 0.349 | 0.199 | 1.75x |
| 128 | 77 | 0.529 | 0.301 | 1.76x |
| 256 | 147 | 1.043 | 0.575 | 1.81x |

结论：

- 把 load permutation 前移到 pack 阶段后，普通 `kGroupGemm` 热路径少了每次 B load 的 `ld_row_N` 重排分支。
- E=32 下新版 LL 单步延迟提升约 1.10x-1.32x。
- E=16 下小并发提升约 1.09x-1.34x，高并发处提升扩大到约 1.75x-1.81x。
- 这说明当前“提前 pack permutation + kernel `STORE_PERMUTE=false`”不仅修正了 operand 顺序，也明显降低了 LL 路径的调度/地址计算开销。

### 2. 固定每个 expert 相同 M

命令：

```bash
cd /workspace/gemm_test
python test_ll_vs_masked.py performance
```

测试条件：`performance` 模式，**warmup=5, measure=20**（每次计时前 5 次预热，取 20 次测量平均）。所有 expert 分配完全相同的 `masked_m = M`，M_alloc = align_up(M, 256)。

配置：`E=32, N=3072, K=3072, CU=128`

| M | LL (ms) | masked (ms) | speedup | LL tokens/s | masked tokens/s | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.309 | 0.335 | 1.08 | 103,683 | 95,457 | LL 略快 |
| 4 | 0.309 | 0.339 | 1.10 | 414,596 | 377,921 | LL 略快 |
| 8 | 0.307 | 0.344 | 1.12 | 832,578 | 743,792 | LL 略快 |
| 16 | 0.319 | 0.350 | 1.10 | 1,605,951 | 1,463,602 | LL 略快 |
| 20 | 0.411 | 0.347 | **0.84** | 1,556,755 | 1,845,905 | **拐点：masked 反超** |
| 32 | 0.411 | 0.352 | 0.86 | 2,490,060 | 2,912,488 | masked 更快 |
| 128 | 1.472 | 0.378 | 0.26 | 2,782,627 | 10,834,259 | masked 明显更快 |
| 1024 | 14.328 | 1.219 | 0.09 | 2,286,972 | 26,881,781 | masked 明显更快 |
| 4096 | 62.497 | 4.771 | 0.08 | 2,097,255 | 27,470,016 | masked 明显更快 |

关键观察：

- **LL 在 M≤16 时一致领先**，speedup 1.08-1.12x。注意 M=1/4/8 时 LL 延迟几乎不变（~0.308 ms），说明 LL kernel 在小 M 下有良好的固定开销控制。
- **拐点在 M≈20**：M=20 时 LL 延迟突然从 0.319 ms 跳到 0.411 ms（+29%），可能命中 LL kernel 内部的 tile 切换阈值；而 masked kernel 延迟在 M=1 到 M=32 全程几乎平坦（0.335-0.352 ms），说明标准 masked 的 M_alloc=256 下调度开销占主导，M 大小不敏感。
- M=16→20 的 LL 延迟跳跃说明 LL kernel 在某个 M 阈值处切换了内部 tiling 策略（从 "小 M 优化" 切换到 "通用路径"），导致单步延迟突然增加。这是 LL kernel 的一个实现特征，不是 bug。
- M≥128 后 LL 延迟随 M 线性增长，而 masked 的 Marlin tile（256×256×128）能充分利用大 M 的并行度，吞吐优势持续扩大。

结论：LL 适合小 `masked_m`（≤16）的 decode 类场景，拐点在 M≈16-20；当每个 expert 都有较大 M（≥32）时，普通 masked 的 Marlin 路径吞吐明显更好。

### 3. D 端 decode 模拟，CU=128

命令：

```bash
cd /workspace/gemm_test
OUTPUT_LEN=1024 CU=128 python test_ll_vs_masked.py decode_d
```

测试条件：`decode_d` 模式，**warmup=10, measure=50**（默认值；可通过环境变量 `WARMUP`/`MEASURE` 覆盖）。每次迭代前 10 次预热 GPU kernel，取 50 次测量平均得到单步延迟。

配置：`E_VALUES=[32, 16], BATCHES=[1,2,4,8,16,32,64,128,256,512], top_k=8, N=3072, K=3072, output_len=1024, CU=128`

说明：当前 `decode_d` 脚本测的是单次 decode step 的 MoE GEMM，`LL (ms)` / `masked (ms)` 仍是单步延迟。`OUTPUT_LEN` 用来估算端到端重复 decode step 的累计 GEMM 时间；脚本会额外打印 `LL total(ms)` / `masked total(ms)`，计算方式是单步平均耗时乘以 `OUTPUT_LEN`。因此把 `OUTPUT_LEN` 从 256 改成 1024，单步延迟理论上应保持一致，累计列近似按 4 倍放大。

#### E=32

| B/conc | routed | active | max_m | M_alloc | LL (ms) | masked (ms) | speedup | LL total(ms) | masked total(ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 1 | 256 | 0.087 | 0.121 | 1.39 | 89.4 | 124.4 |
| 2 | 16 | 13 | 2 | 256 | 0.126 | 0.177 | 1.40 | 128.9 | 180.9 |
| 4 | 32 | 21 | 3 | 256 | 0.189 | 0.267 | 1.41 | 193.6 | 273.0 |
| 8 | 64 | 29 | 4 | 256 | 0.284 | 0.323 | 1.14 | 290.3 | 330.9 |
| 16 | 128 | 31 | 11 | 256 | 0.294 | 0.333 | 1.13 | 301.5 | 340.7 |
| 32 | 256 | 32 | 13 | 256 | 0.301 | 0.341 | 1.13 | 308.5 | 349.4 |
| 64 | 512 | 32 | 21 | 256 | 0.342 | 0.348 | 1.02 | 350.3 | 356.4 |
| 128 | 1024 | 32 | 45 | 256 | 0.483 | 0.349 | 0.72 | 494.8 | 357.4 |
| 256 | 2048 | 32 | 76 | 256 | 0.772 | 0.354 | 0.46 | 790.8 | 362.8 |
| 512 | 4096 | 32 | 148 | 256 | 1.501 | 0.376 | 0.25 | 1536.8 | 384.6 |

#### E=16

| B/conc | routed | active | max_m | M_alloc | LL (ms) | masked (ms) | speedup | LL total(ms) | masked total(ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 1 | 256 | 0.088 | 0.121 | 1.37 | 90.2 | 123.9 |
| 2 | 16 | 12 | 2 | 256 | 0.120 | 0.131 | 1.09 | 122.4 | 133.8 |
| 4 | 32 | 16 | 3 | 256 | 0.147 | 0.187 | 1.28 | 150.2 | 191.5 |
| 8 | 64 | 16 | 6 | 256 | 0.147 | 0.188 | 1.27 | 151.0 | 192.1 |
| 16 | 128 | 16 | 10 | 256 | 0.147 | 0.190 | 1.29 | 150.5 | 194.4 |
| 32 | 256 | 16 | 21 | 256 | 0.183 | 0.194 | 1.06 | 186.9 | 198.3 |
| 64 | 512 | 16 | 40 | 256 | 0.199 | 0.197 | 0.99 | 203.7 | 201.4 |
| 128 | 1024 | 16 | 77 | 256 | 0.301 | 0.197 | 0.65 | 308.6 | 201.9 |
| 256 | 2048 | 16 | 147 | 256 | 0.575 | 0.209 | 0.36 | 589.3 | 214.2 |
| 512 | 4096 | 16 | 282 | 512 | 1.112 | 0.322 | 0.29 | 1138.3 | 329.6 |

结论：

- E=32 时，CU=128 下 LL 在 B=1 到 B=64 都不慢于普通 masked，其中 B=1 到 B=32 有 1.13x-1.41x 优势；B>=128 普通 masked 明显更快。
- E=16 时，CU=128 下 LL 在 B=1 到 B=32 基本持平或更快，其中 B=4/8/16 最明显；B=64 基本持平，B>=128 普通 masked 明显更快。
- 拐点和 `max(masked_m)` 强相关。当前数据里 `max_m<=21` 时 LL 仍有机会；`max_m>=45` 后普通 masked 更稳。

### 4. D 端 decode 模拟，不同 CU 对比

命令：

```bash
cd /workspace/gemm_test
for cu in 64 128 256; do
  CU=$cu BATCHES="1 2 4 8 16 32 64 128 256 512" OUTPUT_LEN=256 TOP_K=8 E_VALUES="32 16" python test_ll_vs_masked.py decode_d
done
```

测试条件：同 `decode_d` 模式，**warmup=10, measure=50**（默认值）。三个 CU 值分别完整跑一轮 benchmark，互不干扰。

下表只列 speedup，便于比较 CU 对选择 LL/masked 的影响。

#### E=32 speedup

| B/conc | max_m | CU=64 | CU=128 | CU=256 | 建议 |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1.45 | 1.39 | 1.37 | LL |
| 2 | 2 | 1.41 | 1.37 | 1.38 | LL |
| 4 | 3 | 1.41 | 1.38 | 1.33 | LL |
| 8 | 4 | 1.16 | 1.13 | 1.09 | LL |
| 16 | 11 | 1.12 | 1.12 | 1.08 | LL |
| 32 | 13 | 1.10 | 1.11 | 1.07 | LL |
| 64 | 21 | 1.04 | 1.01 | 0.94 | 接近持平 |
| 128 | 45 | 0.91 | 0.72 | 0.74 | masked |
| 256 | 76 | 0.72 | 0.46 | 0.42 | masked |
| 512 | 148 | 0.44 | 0.25 | 0.22 | masked |

#### E=16 speedup

| B/conc | max_m | CU=64 | CU=128 | CU=256 | 建议 |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1.38 | 1.38 | 1.34 | LL |
| 2 | 2 | 1.10 | 1.11 | 1.12 | LL |
| 4 | 3 | 1.32 | 1.27 | 1.23 | LL |
| 8 | 6 | 1.32 | 1.26 | 1.24 | LL |
| 16 | 10 | 1.32 | 1.26 | 1.24 | LL |
| 32 | 21 | 1.17 | 1.08 | 1.08 | LL |
| 64 | 40 | 0.95 | 0.98 | 1.03 | 接近持平 |
| 128 | 77 | 0.61 | 0.66 | 0.62 | masked |
| 256 | 147 | 0.42 | 0.37 | 0.38 | masked |
| 512 | 282 | 0.40 | 0.28 | 0.33 | masked |

结论：

- CU=64 在 E=32 的大多数 decode 点上更稳，尤其 B>=128 时 LL 退化较慢，但仍不如普通 masked。
- CU=128 是当前默认测试值，小 B decode 收益明确。
- CU=256 没有带来稳定收益；E=32 的 B=64 已经低于 1，E=16 的 B=64 仅小幅高于 1。
- 对集成策略来说，CU 不是唯一判断条件，`max(masked_m)` 和并发更关键。





新deepgemm包位置

/zkjh/packages/deepgemm-2.1.0+das.dtk2604.masked_ll.e52469a-cp310-cp310-linux_x86_64.whl



肖渝老师gfx938

![image-20260608101856444](C:\Users\lesheng\AppData\Roaming\Typora\typora-user-images\image-20260608101856444.png)

gfx936测试

![image-20260608101941245](C:\Users\lesheng\AppData\Roaming\Typora\typora-user-images\image-20260608101941245.png)





## 集成建议

### 拐点精确定位

结合固定 M micro benchmark（本节第 2 部分）和 D 端模拟（第 3/4 部分），LL vs masked 的拐点如下：

| 数据来源 | LL 优势区间 | 持平区间 | masked 优势区间 |
|----------|------------|---------|----------------|
| 固定 M（每个 expert 相同 M） | M ≤ 16 | M ≈ 16-32 | M ≥ 32 |
| D 端模拟（随机路由分布） | max_m ≤ 21 | max_m ≈ 21-45 | max_m ≥ 45 |

两种测试的拐点不同，原因是：

- **固定 M 测试**：所有 expert 的 M 相同，是最严苛的条件。M=20 时 LL 内部触发 tile 切换，延迟跳变 → 被 masked 反超。
- **D 端模拟**：router 分布不均，虽然 `max_m` 可能到 21，但大多数 expert 的 M 远小于 `max_m`。LL 在小 M expert 上的优势抵消了大 M expert 上的劣势，因此 `max_m=21` 时 LL 仍可能总体领先。

**实际应用时以 D 端模拟（随机路由分布）为准**，因为它更接近真实 decode 场景。

### 保守启发式

```text
decode 场景 && max(masked_m) <= 16:
    优先尝试 m_grouped_w8a8_gemm_nt_masked_ll
否则:
    优先使用 m_grouped_w8a8_gemm_nt_masked
```

### 更细的启发式（基于 D 端模拟数据）

```text
E=32:
    B/conc <= 32 时 LL 通常更快。
    B/conc = 64 时接近持平，是否使用 LL 取决于真实 router 分布和 CU。
    B/conc >= 128 时普通 masked 更好。

E=16:
    B/conc <= 32 时 LL 通常更快。
    B/conc = 64 时接近持平。
    B/conc >= 128 时普通 masked 更好。
```

### 注意事项

- LL kernel 在 M 越过内部阈值时会触发 tile 策略切换，延迟有一次跳变（~0.31 ms → ~0.41 ms），这不是精度问题而是实现特征。如果端到端中发现 LL 延迟在某个 batch size 突然变差，检查 `max(masked_m)` 是否越过了 16→20 的阈值。
- 上述启发式基于 E ∈ {16, 32} 的 D 端模拟。如果实际 E 不同（如 E=1），需要重新测试。

后续接入模型服务时，建议继续在真实 router 输出的 `masked_m` 分布上测，而不是只看随机分布或固定 M micro benchmark。重点记录：

- `batch_size` / 并发
- `top_k`
- `E`
- active expert 数
- `max(masked_m)`
- LL 耗时
- 普通 masked 耗时
- generate throughput / total throughput
- TPOT

## decode_d 参数解释

`test_decode_d_performance()` 在 [test_ll_vs_masked.py:326](gemm_test/test_ll_vs_masked.py#L326) 定义，模拟 **D 端 decode 阶段**的 MoE GEMM 行为。

### 核心概念：D 端 decode 的特点

在 MoE 推理中，decode 阶段每个 step 每层只生成 **1 个 token per request**。这个 token 经 router 分配给 top_k 个 expert。因此：

- **每个 expert 分到的 token 数（masked_m[e]）通常很小**（等于并发请求中被路由到该 expert 的数量）
- **大量 expert 可能分到 0 个 token**（尤其 E=32、并发不高时）
- M_alloc = align_up(max(masked_m), 256) 决定了 kernel 的 M 维度

这就是 LL kernel 的用武之地：**小 M 场景下 LL 的调度开销更低**。

### 各参数含义

| 参数 | 环境变量 | 默认值 | 含义 |
|------|----------|--------|------|
| `E_VALUES` | `E_VALUES` | `[32, 16]` | 专家数列表，每轮测一个 E |
| `BATCHES` | `BATCHES` | `[1,2,4,8,16,32,64,128,256,512]` | 并发请求数（= batch_size），每步每请求 1 token |
| `top_k` | `TOP_K` | `8` | 每个 token 路由到的 expert 数 |
| `K` | `K` | `3072` | hidden 维度 |
| `N` | `N` | `3072` | intermediate 维度 |
| `output_len` | `OUTPUT_LEN` | `256` | 模拟的输出长度，用于估算累计 GEMM 时间（不改变单步行为） |
| `cu` | `CU` | `128` | LL kernel 的 CU 参数（compute unit 数） |
| `WARMUP` | `WARMUP` | `10` | warmup 迭代次数 |
| `MEASURE` | `MEASURE` | `50` | 测量迭代次数 |

### 输出列含义

| 列 | 含义 |
|----|------|
| `B/conc` | 并发请求数（= batch_size），即同时 decode 的请求数 |
| `routed` | 总路由 token 数 ≈ B × top_k（实际可能因采样有微小差异） |
| `active` | 被至少 1 个 token 命中的 expert 数 |
| `max_m` | 所有 expert 中分到 token 数最多的那个 expert 的 token 数 |
| `M_alloc` | `align_up(max(max_m, 1), 256)`，kernel 实际分配的 M 维度 |
| `LL (ms)` | LL kernel 单步 MoE GEMM 延迟（50 次平均） |
| `masked (ms)` | 标准 masked kernel 单步延迟 |
| `speedup` | `masked(ms) / LL(ms)`，>1 表示 LL 更快 |
| `LL total(ms)` | `LL(ms) × output_len`，估算 output_len 步的累计 LL GEMM 时间 |
| `masked total(ms)` | `masked(ms) × output_len` |
| `LL step tok/s` | `B / (LL(ms)/1000)`，单步吞吐（requests/s） |
| `masked step tok/s` | `B / (masked(ms)/1000)` |

### 关键参数：M_alloc

`M_alloc` 是性能的核心决定因素。在 decode 场景下：

```
M_alloc = align_up(max(masked_m), 256)
```

- `max(masked_m)` ≈ B × top_k / E × (负载不均系数)，通常在 1~几十
- 当 B ≤ 32、E=32 时，max(masked_m) 通常在 1~21，M_alloc=256
- 当 B ≥ 128 时，max(masked_m) 可能超过 45，M_alloc≥256

**LL kernel 在小 M_alloc 下有优势**（调度快、padding 少），**标准 masked kernel 在大 M_alloc 下有优势**（Marlin tile 并行度更高）。

## LL vs Masked 的优势与不足

### LL kernel 优势

1. **低延迟调度**：LL kernel 针对小 M 优化，grid launch 开销更低。当 `max(masked_m) ≤ 21` 时，单步延迟显著优于标准 masked。
2. **权重 layout 紧凑**：6-D packed weight（W6 layout）比 Marlin layout 在 decode 场景的内存访问模式更友好。
3. **auto-tuning**：LL kernel 内部有 `find_m_block` 等 auto-tuning 逻辑，根据实际 `masked_m` 动态选择最优 tile 配置。
4. **专家数弹性**：E=16/32 均已验证，对小 EP 场景覆盖良好。

### LL kernel 不足

1. **大 M 退化严重**：当 `max(masked_m) ≥ 45` 时，标准 masked 的 Marlin 路径吞吐优势明显（speedup < 0.5）。原因：LL kernel 的 tile 并行度受限于小 M，而 Marlin 可以用更大的 M tile 摊销调度开销。
2. **维度限制**：LL kernel 的 (E, N, K) 均有硬编码限制：
   - E ∈ {1, 16, 32}
   - N ∈ {3072, 4096, 6144, 7168}
   - K ∈ {1536, 2048, 3072, 6144, 7168}
   不符合这些维度的模型无法使用 LL 路径。
3. **不支持 block_wise=True**：当前 LL 路径只在 `block_wise=False`（2-D scales）下验证通过。
4. **CU 参数不敏感**：CU=64/128/256 对 speedup 的影响有限，无法通过调 CU 显著扩大 LL 的适用范围。

### 标准 Masked kernel 优势

1. **大 M 高吞吐**：当 M ≥ 128 时，Marlin tile 256×256×128 的并行度充分发挥，吞吐远超 LL。
2. **无维度限制**：支持任意 E、N、K 组合（只要显存够）。
3. **支持 block_wise**：block_wise=True 时用 3-D scales，精度更高。
4. **CUDA graph 兼容**：支持 `--cuda-graph` 模式（LL 路径暂未验证）。

### 标准 Masked kernel 不足

1. **小 M 调度开销大**：grid launch 和 Marlin tile 初始化开销在 M 很小时占比高。
2. **Grid 计算复杂**：之前有过 grid 硬编码 bug（gdx=128 硬编码），需要正确计算 `globalWorkSize`。

## 实际 Decode 例子

以 MiniMax-M2.5 的典型 decode 场景为例：TP=8, EP=8, E=32, top_k=8, N=3072, K=3072。

### Step-by-step 推演

假设当前有 **B=16 个并发请求**正在 decode。每个 step：

```
Step 1: 每请求生成 1 个 token → 共 16 个 token
Step 2: Router 将每个 token 分配给 top_k=8 个 expert
        → 总计 16×8 = 128 次路由
        → 分布在 32 个 expert 上
```

Router 可能的分配结果（随机示例）：

```
expert  |  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 ...
tokens  |  5  3  4  2  6  1  4  3  5  2  4  3  6  2  3  4 ...
```

实际 `masked_m` 数组（部分）：

```python
masked_m = [5, 3, 4, 2, 6, 1, 4, 3, 5, 2, 4, 3, 6, 2, 3, 4,
            4, 5, 3, 2, 5, 1, 4, 3, 5, 2, 4, 3, 5, 2, 3, 4]
max(masked_m) = 6
M_alloc = align_up(6, 256) = 256
active = 32  (每个 expert 至少 1 个 token)
routed = 128
```

### 两种 kernel 的行为差异

**LL kernel**：
```
M_alloc=256, 每个 expert 只计算 masked_m[e] 行（5 行、3 行...）
→ 有效 M 很小，LL 的低延迟调度优势明显
→ 单步延迟 ≈ 0.147 ms（E=16）/ 0.294 ms（E=32）
→ 单步吞吐 ≈ 16/0.000294 ≈ 54,400 req/s
```

**标准 Masked kernel**：
```
M_alloc=256, 每 expert 计算 masked_m[e] 行
→ Marlin tile 256×256×128 的 M 维度只用了一小部分
→ 调度和 tile 初始化开销占比高
→ 单步延迟 ≈ 0.333 ms（E=32）
→ 单步吞吐 ≈ 16/0.000333 ≈ 48,000 req/s
```

**LL 领先约 1.13x**。

### 如果 B 增大到 256

```python
# B=256, E=32, top_k=8
routed = 256×8 = 2048
avg_per_expert = 2048/32 = 64
max(masked_m) ≈ 76 (考虑负载不均)
M_alloc = align_up(76, 256) = 256

# 但此时标准 masked 的 Marlin tile 更好地利用了 M=76 的并行度
LL:   0.772 ms  (单步)
masked: 0.354 ms  (单步)
speedup = 0.354/0.772 = 0.46  → masked 明显更快
```

### 结论

**拐点在 `max(masked_m)` ≈ 21**：
- `max_m ≤ 21` → LL kernel 更快
- `max_m ≥ 45` → 标准 masked 更快
- `21 < max_m < 45` → 接近持平，取决于 E 和 CU

对应到并发数（E=32, top_k=8）：
- B ≤ 32 → LL 通常更快
- B = 64 → 接近持平
- B ≥ 128 → 标准 masked 更快

## 端到端测试脚本

项目中两个 shell 脚本构成了完整的端到端测试链路：`minimax_masked_tp8_ep8.sh` 启动服务端，`d_bench.sh` 运行压测客户端。

### 1. minimax_masked_tp8_ep8.sh — 服务端启动脚本

**路径**：[gemm_test/sh_file/minimax_masked_tp8_ep8.sh](gemm_test/sh_file/minimax_masked_tp8_ep8.sh)

```
启动 → sglang.launch_server → MiniMax-M2.5-W8A8 → 等待客户端请求
```

**关键配置**：

| 配置 | 值 | 说明 |
|------|-----|------|
| 模型 | `/zkjh/weight/MiniMax-M2.5-W8A8` | W8A8 量化后的 MiniMax-M2.5 |
| TP=8, EP=8 | tensor parallel 8 + expert parallel 8 | 共 8 张 DCU |
| DP=1, PP=1 | 不拆分 data/pipeline | 单节点推理 |
| `--mem-fraction-static 0.9` | 90% 显存给 KV cache | 留 10% 给权重和临时张量 |
| `--chunked-prefill-size 32768` | 32768 token chunked prefill | 长 prompt 分块处理 |
| `--max-running-requests 256` | 最多 256 并发请求 | 控制显存和调度压力 |
| `--attention-backend fa3` | FlashAttention 3 | DCU 上的高效 attention |
| `--moe-a2a-backend deepep` | DeepEP 做 all-to-all | expert parallel 的通信后端 |
| `--minimax-opt` | MiniMax 专用优化 | 可能包含 router 优化等 |
| `ROCSHMEM_*` 环境变量 | 288 QP / 48 context / 3.7G heap | DeepEP 的 ROCSHMEM 配置 |

**在 MoE GEMM 层面的作用**：
- sglang 启动后，每层 MoE 的 forward 会调用 `m_grouped_w8a8_gemm_nt_masked` 或 `m_grouped_w8a8_gemm_nt_masked_ll`
- 当前 sglang 传 `config={'MODE': 1000}` → 走标准 masked 路径（ASM 256×256×128）
- 如果要切到 LL 路径，需要修改 sglang 的 `ep_moe/layer.py` 调用处

### 2. d_bench.sh — 压测客户端脚本

**路径**：[gemm_test/sh_file/d_bench.sh](gemm_test/sh_file/d_bench.sh)

```
启动 → sglang.bench_serving → 发请求 → 收集指标 → 写入 CSV
```

**关键参数**：

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| 模型路径 | `MODEL_PATH` | `/models/docker_data/MiniMax-M2.5` | 用于生成请求格式 |
| 服务端口 | `PORT` | `30002` | sglang 服务端口 |
| 并发数 | `BATCHES` | `8` | 可传空格分隔的多个值，如 `"8 16 32"` |
| 输出长度 | `GSP_OUTPUT_LEN` | `1024` | 每个请求生成的 token 数 |
| 输入长度 | `TOTAL_INPUT_TOKEN` | `4096` | 每个请求的输入 token 数 |
| prefix cache 占比 | `PREFIX_CACHE` | `99` | 99% 输入共享（模拟 system prompt cache） |
| request rate | `rr` | `inf` | 无限速率（一次性全发） |
| 并发倍数 | `CONCURRENCY_MULTIPLIER` | `1` | prompts_per_group = batch × multiplier |

**收集的关键指标**：

| 指标 | 含义 | 与 MoE GEMM 的关系 |
|------|------|-------------------|
| `generate_throughput_tok_s` | 生成吞吐（tok/s） | **直接受 decode GEMM 延迟影响** |
| `total_throughput_tok_s` | 总吞吐（含 prefill） | prefill + decode 综合 |
| `mean_tpot_ms` | 平均每 token 输出时间 | **decode 延迟的端到端体现** |
| `p95_tpot_ms` / `p99_tpot_ms` | TPOT 的 P95/P99 | 反映长尾延迟 |
| `mean_ttft_ms` | 平均首 token 时间 | 主要受 prefill 影响 |
| `mean_itl_ms` | 平均 token 间延迟 | 细粒度 decode 延迟 |
| `request_rate` | 请求吞吐（req/s） | 整体服务能力 |

### 测试流程

```text
1. 启动服务端
   bash minimax_masked_tp8_ep8.sh
   → 等待模型加载完毕，监听 8000 端口

2. 运行压测（另一个终端）
   BATCHES="8 16 32 64 128 256" bash d_bench.sh
   → 依次对每个并发数跑一轮 benchmark
   → 每轮输出到 client/<model>-<date>/ 目录
   → 汇总到 all.csv

3. 分析结果
   - B=8 时 TPOT 应该在几十 ms 量级
   - 随着 B 增大，TPOT 上升（GPU 算力饱和）
   - 如果 B≥128 时 TPOT 明显恶化，可能是 decode GEMM 瓶颈
   - 此时切 LL kernel 可能有收益（如果 max_m 仍在 ≤21 范围）
```

### 从 micro benchmark 到端到端的映射

`test_ll_vs_masked.py decode_d` 测的"单步延迟"和 `d_bench.sh` 的 `mean_tpot` 之间的关系：

```
mean_tpot ≈ decode_GEMM_time + attention_time + all_to_all_time + other_overhead
           ≈ LL(ms) + (attention + a2a + misc)
```

其中 `decode_GEMM_time` 就是 `LL(ms)` 或 `masked(ms)`。在 EP=8 且 all-to-all 优化良好的情况下，GEMM 可能占总 TPOT 的 30-60%。

**示例估算**（B=16, E=32, output_len=1024）：
```
LL kernel:  单步 GEMM = 0.294 ms, TPOT 假设 ≈ 0.8 ms
masked:     单步 GEMM = 0.333 ms, TPOT 假设 ≈ 0.85 ms

端到端 generate throughput:
LL:     16 requests × 1024 tokens / (1024 × 0.8 ms) ≈ 20.0 tok/s/request → 320 tok/s total
masked: 16 requests × 1024 tokens / (1024 × 0.85 ms) ≈ 18.8 tok/s/request → 301 tok/s total

LL 端到端吞吐提升约 6%
```

如果切到 LL kernel，需要注意：
1. 确认模型维度在 LL kernel 的支持范围内（MiniMax-M2.5 的 E=32, N=3072, K=3072 满足）
2. sglang 的 `ep_moe/layer.py` 需要修改调用接口（LL 用 2-D scales + W6 weight）
3. 在真实 router 分布下验证 TPOT 变化，而不是只看 micro benchmark
