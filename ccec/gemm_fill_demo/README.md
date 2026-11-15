# gemm_fill_demo

带 **L1 补零**（`asc_fill_l1`）的纯 ccec c_api CUBE GEMM 样例：

```
C[M,N] = A[M,K] x B[K,N]      half x half -> float
M = 64, N = 128, K = 208, L1 内 K 方向按 K_TILE = 256 分配
```

`K = 208` 只占 13 个 16 列分形，但 L1 的 k 分形表按 16 个分配（多出来的 3 个 k 分形是
"补边区"）。k 循环固定跑满 16 拍，**补边区必须先用 `asc_fill_l1` 清零**，否则 mmad 会把
残留/越界数据乘进 C。

> **本文件已迁移到 CANN 9.2.1**：不再使用 9.2.0 起弃用的 `asc_fill_l1_sync`，并顺带把
> `nd2nz / mmad / l0c2gm` 换成 9.2.1 的推荐重载、把热循环里的 `PIPE_ALL` 屏障换成
> cube 侧事件对。详见 [CANN 9.2.1 迁移](#cann-921-迁移asc_fill_l1_sync-弃用)。

本样例给 4 个变体，用实测误差反推"补零到底补在哪、补多少"：

| 变体 | GM→L1 搬入宽度 | `asc_fill_l1` 补几个 k 分形 | 预期 | 说明 |
| --- | --- | --- | --- | --- |
| v0 | 208（只搬有效列） | 3 | PASS（max err = 0） | 窄搬入 + 补零，等价于"尾部完全清零" |
| v1 | 256（连毒化尾巴一起搬） | 0 | FAIL（≈ 480000） | 不补零，毒化数据被累加 |
| v2 | 256 | 3 | PASS（max err = 0） | 全宽搬入 + 补零，补零把尾巴盖掉 |
| v3 | 256 | 2（少补一个分形） | FAIL（≈ 160000） | 少补 16 列，误差正好正比于漏补列数 |

GM 里 `k >= 208` 的尾部统一填 `100.0`（half 精确可表示），所以"漏补 1 个 k 分形
(16 列)"的误差恰好是 `100 * 100 * 16 ≈ 160000`，数量级一眼可辨。

## 文件结构

| 文件 | 说明 |
| --- | --- |
| `gemm_fill.asc` | kernel（4 个变体）+ host 侧 ACL 全流程 + CPU 三重循环校验 + cycle 打印 |
| `CMakeLists.txt` | ASC CMake 构建脚本，目标架构 `dav-3510`，运行模式 `npu` |

## 构建与运行

需要 **CANN >= 9.2.1**（本样例用到 9.2.1 的强类型枚举重载与 cube 侧事件对）：

```bash
source /home/xinyang/CANN/9.2.1/ascend-toolkit/set_env.sh
cmake -B build -DCMAKE_ASC_RUN_MODE=npu -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build
./build/demo [device_id]
```

> 运行依赖 Ascend NPU（`aclrtSetDevice` 需要有效 devId）。无设备的开发机只能验证编译，
> 本样例在无设备时会直接打印 `aclrtSetDevice ... failed` 并返回 1，不会再去读设备内存。

预期输出（有设备的机器上）：

```
GEMM [64x208] x [208x128]  (half x half -> float), L1 K 方向按 256 补齐
补零参数: repeat=4, blk_num=(K1_TILE-K1_VALID)*16=48, dst_gap=K1_VALID*16=208
尾部毒化值 = 100，漏补 t 个 k 分形时误差约 10000 * 16 * t

v0 窄搬入(208) + 补3分形
  max abs error = 0 ...
  result: PASS(=参考值)  [预期: PASS -> OK]

v1 全宽搬入(256) + 不补零
  max abs error = 480000 ...
  result: MISMATCH(与参考值不一致)  [预期: MISMATCH -> OK]
...
overall: ALL AS EXPECTED
```

## CANN 9.2.1 迁移（`asc_fill_l1_sync` 弃用）

### 1. 弃用信息

`c_api/composite/cube_datamove_composite.h` 里：

```cpp
ASC_DEPRECATED(9.2.0, "2027/09/07", "asc_fill_l1 and asc_sync")
__aicore__ inline void asc_fill_l1_sync(__cbuf__ half* dst, half value,
                                        const asc_fill_value_config& config);
```

`_sync` 变体的实现是 `create_cbuf_matrix(...) + asc_sync_post_process()`，而
`asc_sync_post_process()` 在 9.2.1 里仍是 `pipe_barrier(PIPE_ALL)`。也就是说 **A、B 各一条
补零会各自触发一次全流水线屏障**。9.2.1 的推荐写法是"异步指令 + 显式同步"。

### 2. 迁移前后的写法

```cpp
// ---- 9.1：每条补零自带全屏障 ----
asc_fill_l1_sync(a_l1 + pad_off, half(0), cfg);   // 内部 pipe_barrier(PIPE_ALL)
asc_fill_l1_sync(b_l1 + pad_off, half(0), cfg);   // 又一次

// ---- 9.2.1：异步发射，一条同步收口 ----
asc_fill_l1(a_l1 + pad_off, half(0), cfg);
asc_fill_l1(b_l1 + pad_off, half(0), cfg);
asc_copy_gm2l1_nd2nz(...);        // 同一条 MTE2 上也已发射
asc_sync();                       // 一次收口：MTE2 搬运 + 补零 全部完成
```

`asc_sync()` 放在"每个变体的搬入 + 补零"之后，不在热循环里，所以它只发生 4 次（每变体 1
次），而原来光是补零就 8 次（A、B 各 4 次）全屏障。

### 3. 热循环同步也一起换成 9.2.1 的 cube 侧事件对

9.1 的 3510 `set_flag/wait_flag` 只接受 `PIPE_S/PIPE_V` 与 `PIPE_MTE2/PIPE_MTE3`，
`pipe_barrier` 只接受 `PIPE_MTE2/PIPE_MTE3/PIPE_ALL`，所以之前 cube 侧只能
`asc_sync()`（每拍 2 次 `PIPE_ALL` 屏障，512 拍共 1024 次）。9.2.1 的
`asc_sync_notify/asc_sync_wait` 在 AIC 上显式支持这些配对
（`impl/c_api/reg_base_impl/sync_intf_impl.h` 的 `asc_sync_notify_impl_aic`）：

| 生产者 → 消费者 | 用途 |
| --- | --- |
| `PIPE_MTE2 → PIPE_MTE1` | GM→L1 / 补零 完成后再读 L1（本例用 `asc_sync()` 收口） |
| `PIPE_MTE1 → PIPE_M` | L1→L0A/L0B 完成后再 mmad |
| `PIPE_M → PIPE_MTE1` | mmad 读完 L0A/L0B 后下一拍才能覆盖 |
| `PIPE_M → PIPE_FIX` | mmad 结果对 FIX 可见后再 L0C→GM |
| `PIPE_FIX → PIPE_M` | L0C 写完再让下一拍 mmad 覆盖 L0C |

于是热循环里每拍只剩 2 次细粒度 `set_flag/wait_flag`，不再有整条流水线屏障。

### 4. 顺带迁掉的其他弃用接口

| 9.1 写法 | 9.2.1 推荐写法 | 说明 |
| --- | --- | --- |
| `asc_copy_gm2l1_nd2nz(..., uint8_t l2_cache_ctl, ...)` | `asc_copy_gm2l1_nd2nz(..., asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, ...)` | 第 3 参改名为 `src_d_value`（仍是源行步长，字节），L2 模式变强类型枚举 |
| `asc_mmad(c, a, b, 16,16,16, 0, false, ...)` | `asc_mmad(c, a, b, m, k, n, asc_unit_flag_mode::DISABLE, false, ...)` | 形参名改为 `m/k/n`，unit flag 用枚举 |
| `asc_copy_l0c2gm(..., 21 个参数)` | `asc_copy_l0c2gm(dst, src, n_size, m_size, dst_stride, src_stride, asc_store_l2_cache_mode, asc_unit_flag_mode, asc_quant_mode, asc_relu_pre_mode, channel_split, nz2nd, nz2dn, clip_relu_pre)` | 21 参 → 14 参，量化/relu 用枚举 |
| `asc_copy_l12l0b_trans(..., dst_gap, 0, 0)` | 显式 `static_cast<uint16_t>(0), static_cast<uint16_t>(0)` | 见下方"坑" |

编译后 **0 warning**（`-Wdeprecated-declarations` 全部清零）。

### 5. 9.2.1 的坑：`l12l0b_trans` 新重载只有声明没有实现

9.2.1 头文件里有两个参数个数相同的重载：

```cpp
// 新：只有声明，.cube 里没有实现 —— 链接报 undefined symbol
asc_copy_l12l0b_trans(dst, src, index_id, repeat, src_stride, dst_gap,
                      bool enable_addr_decrement, uint16_t dst_frac_gap);
// 旧：有实现（9.1 起就在用）
asc_copy_l12l0b_trans(dst, src, index_id, repeat, src_stride, dst_gap,
                      uint16_t dst_frac_gap, uint16_t src_frac_gap);
```

直接写 `(..., 0, 0)` 会命中"二义性"（`0` 可同时转 `bool` 和 `uint16_t`）；改成
`(..., false, 0)` 会选中新重载，然后链接失败：

```
ld.lld: error: undefined symbol: asc_copy_l12l0b_trans(half AS4*, half AS2*, unsigned short,
          unsigned char, unsigned short, unsigned short, bool, unsigned short) (.cube)
```

所以本样例显式传两个 `uint16_t` 的 0，落到有实现的那个重载上。

## `asc_fill_l1` 的实现链路（读 CANN 源码核对）

### 1. 接口与参数

`c_api/cube_datamove/cube_datamove.h`：

```cpp
__aicore__ inline void asc_fill_l1(__cbuf__ half* dst, half value,
                                   const asc_fill_value_config& config);
```

`c_api/defs/union.h`（9.2.1 里从 `c_api/utils/union.h` 挪到了 `defs/`）中
`asc_fill_value_config` 是 64bit 打包结构：

| 位域 | 含义 |
| --- | --- |
| `[14:0]` `repeat` | 重复次数 |
| `[30:16]` `blk_num` | 每次重复填充的 **32B block** 数（连续） |
| `[46:32]` `dst_gap` | 两次重复之间**跳过**的 32B block 数 |

即：`for r in [0, repeat): fill [dst + r*(blk_num+dst_gap)*32B, + blk_num*32B) = value`，
`value` 按元素类型广播（half 时一个 32B block = 16 个元素）。

### 2. 落到哪条硬件指令

`impl/c_api/memory_base_impl/cube_datamove_composite_intf_impl.h`（9.2.1）里，
每个 `asc_fill_l1_impl` 都是

```cpp
if ASC_IS_AIC { create_cbuf_matrix(dst, config.config, value); }
```

`create_cbuf_matrix` 就是 CCE 的"给 L1(cbuf) 批量写常量"指令
（`cce_aicore_intrinsics.h`: `__builtin_cce_create_cbuf_matrix`，bf16 变体
`create_cbuf_matrix_bf16`、`create_cbuf_matrix_h`、`create_cbuf_matrix_ui`）。

### 3. 语义核对（CANN 里的两个生产用例）

- `impl/tensor_api/.../TFillPad.hpp` 的 `TFillPad_cube`：清零 NZ 图里"有效行之外的行"时，
  把 `blockLen(行数)` 写进 `[30:16]`、`repeat`（k 方向 32B block 数）写进 `[14:0]`、
  `repeatGap` 写进 `[46:32]`，与上面的地址公式完全一致：

  ```cpp
  int64_t repeatConfig = (blockLen << 16) | (repeatGap << 32) | repeat;
  pto_create_cbuf_matrix(dst, repeatConfig, 0);
  ```

- blaze GEMM 的 K 方向补零（`.../blaze/gemm/tile/arch35/pad_mx_kl1.h`，arch35 = 3510）：
  做法正是 `asc_fill_l1((__cbuf__ half*)tensorL1.data().get(), half(0), config)`，
  `repeat=m1`、`blk_num=kAxisL1Align-kAxis`、`dst_gap` 按 L1 行跨度算 —— 与本样例
  `pad_zero_k_tail()` 的三参数一一对应。9.2.1 里它同样**只用异步 `asc_fill_l1`**，
  可见"异步 + 外部统一同步"就是当前的推荐姿势。

## L1 的 NZ 布局与补零参数推导

`nd2nz` 的目标分形布局由 `asc_set_gm2l1_nz_para(MTE2_NZ_PARA)` 决定
（`impl/tensor_api/arch/cube/gm_to_l1/copy_impl/nd2nz.h` 里 `mte2NzPara` 的位域）：

```
[15:0] nd_num | [31:16] dst_nz_n_stride | [47:32] dst_nz_c0_stride | [63:48] dst_nz_matrix_stride
```

本样例设成 `nd_num=1, n_stride=1, c0_stride=16, matrix_stride=0`，对应
`[m1][k1][16][16]`（k1 在最内层）的 NZ 布局：

```
elem(m, k) 偏移 = (m/16)*K1*256 + (k/16)*256 + (m%16)*16 + (k%16)
=> 第 (mf, kf) 个分形的起始元素偏移 = (mf*K1 + kf) * 256
=> 一个 16x16 half 分形 = 512B = 16 个 32B block
```

于是"k >= 13 的补边区"在每个 m 分形内是连续的 3 个分形 = 48 个 32B block；相邻 m
分形之间要跳过 13 个有效分形 = 208 个 32B block。补零参数就取：

```cpp
repeat  = M/16 = 4
blk_num = (K1_TILE - K1_VALID) * 16 = 3*16 = 48      // 3 个 k 分形 = 48 个 32B block
dst_gap = K1_VALID * 16            = 13*16 = 208     // 跳过 13 个有效分形
```

自检：重复步进 `(48+208)*32B = 8192B = 4096 half`，正好是一个 m 分形的跨度
`K1_TILE*256 = 16*256 = 4096` ✓

## 数据流与同步

```
GM --nd2nz--> L1(NZ) --l12l0a/b--> L0A/L0B --mmad--> L0C --l0c2gm--> GM
        MTE2        MTE1              M              FIX
```

- `asc_copy_gm2l1_nd2nz(dst, src, src_d_value=K_TILE*sizeof(half), l2_mode, n_value=M/N,
  d_value=copy_k, src_nd_matrix_stride=0, enable_small_c0=false)`；
- `asc_copy_l12l0a(a_l0, a_frac, 0, 0, 1, 1, 1, 1)`：单分形，指针已定位到分形起点；
- B 在 GM 里是转置 `[N,K]`，L1 中同样是 `[n,k]` 分形，进 L0B 要转置（Zn）；
- `asc_mmad(c_l0, a_l0, b_l0, 16, 16, 16, asc_unit_flag_mode::DISABLE, false,
  kf>0/*累加*/, kf==0/*首拍清零*/)`；
- `asc_copy_l0c2gm(c_tile, c_l0, n_size=16, m_size=16, dst_stride=N, src_stride=1,
  asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, asc_unit_flag_mode::DISABLE,
  asc_quant_mode::NoQuant, asc_relu_pre_mode::NONE, false, true/*nz2nd*/, false, false)`。

## 实证状态（重要）

- **编译**：CANN `9.2.1`（`dav-3510`）下 `cmake --build` 通过，**0 warning**
  （9.1 下的 5 条 `-Wdeprecated-declarations` 全部消除）。
- **补零几何（无需 NPU）**：`main()` 在 `aclInit` 之前跑一个 host 侧模型，按
  `repeat/blk_num/dst_gap` 复现 `create_cbuf_matrix` 的地址游走，检查覆盖的 32B block
  集合是否正好等于"所有 `kf >= 13` 的 block"。本机实测输出：

  ```
  [fill-geometry self-check] A tile: 1024 个 32B block, 补零覆盖 192 个 (期望 192 = M/16 * 3分形 * 16block), mismatch = 0 -> OK
  [fill-geometry self-check] m-fractal 0 覆盖图: .............###   (. 有效 k 分形 13 个 / # 补零 k 分形 3 个)
  ```

  即参数推导（48/208/4）与 `[m1][k1][16][16]` 布局自洽。
- **运行**：本机（WSL）无 `/dev/davinci*`，`aclInit` 失败（ret 500000），只验证到
  "host 侧流程正常、kernel 能编译并注册"。4 个变体的 PASS/FAIL 预期需要
  **在有 3500 系列 NPU 的机器上**跑 `./build/demo` 才能确认。
- **同步改动的影响**：热循环从"每拍 2 次 `PIPE_ALL` 屏障"改成"每拍 2 次
  `set_flag/wait_flag`"，理论上是净收益；但在真实硬件上如果出现偶发错数，优先怀疑
  MTE1/M/FIX 的 flag 复用（`EVENT_ID0` 被多组事件对共用）——可以给每组配独立
  `event_t` 再验一遍。

## 与 `gemm_demo` 的关系

`../gemm_demo` 是同一 kernel 结构的**无补零**版本（K=512，16 对齐，不需要补零），
可以对照着看"有/没有补零"的差别。它仍是 9.1 的写法（9.2.1 下可编译但会有弃用告警）。
