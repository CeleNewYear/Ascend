# pltm_demo

Ascend950（DAV_3510）**PLTM** 指令验证与演示样例：用 ccec 内建 `pltm_b32(offset, ub)`
生成运行时 predicate mask，做一次 predicated 翻倍，验证输出为"前 x 个 2、其余 1"。
与 `../plt_demo` 验证口径完全一致，**x 同样是运行时动态值**；区别仅在于 mask 生成指令
从有状态的 PLT 换成无状态的 PLTM。

## 文件结构

| 文件 | 说明 |
| --- | --- |
| `pltm_demo.asc` | kernel（`pltm_b32` + `vlds`/`vmuls`/`vsts`）与 host 入口，ASC 单文件工程 |
| `data_utils.h` | 输入/输出文件读写辅助工具 |
| `CMakeLists.txt` | ASC CMake 构建脚本，目标 `dav-3510` |
| `scripts/gen_data.py` | 生成全 1 输入、运行时 x、golden（前 x 个 2） |

## 构建与运行

```bash
python3 scripts/gen_data.py                       # x 随机；固定值：--x 150
cmake -B build -DCMAKE_ASC_RUN_MODE=npu  -DCMAKE_ASC_ARCHITECTURES=dav-3510   # 真机
cmake -B build -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510   # 模拟器
cmake --build build
./build/demo
```

预期输出：`runtime x (active_count) = <x>`；`test pass!`（与 `plt_demo` 相同的 256 元素验证）。
本样例已在 dav-3510 模拟器（sim 模式）实测：x=150 与 x=200 均 `test pass!`。

> host→device 数据拷贝请使用 `ACL_MEMCPY_HOST_TO_DEVICE`（与 `cycle_count_demo` 一致）；
> 模拟器环境下 `ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE` 不生效，输出会全 0。
>
> 实现约束：`__simd_vf__` 内只使用 for 直落循环（while/if 条件控制流 + 向量指令会触发
> bisheng dav-c310-vec 后端段错误，见 `../plt_demo/README.md`）；PLTM 的双参数（offset/ub）
> 恰好让"每帧 mask"可以无分支地由循环变量直接给出。

## PLTM 指令语义讲解

`pltm_b32(offset, ub)` 两个输入都是**只读**标量：

- `offset`（uint16）= **当前这一帧之前已经处理完的整帧数**（帧号，0 起）。
  本样例循环变量 `frame` 直接充当 offset —— 第 f 帧之前已处理 f×64 个元素。
- `ub`（uint32）= 待处理元素总数的**上界**。本样例即运行时 x。

生成 mask 的硬件公式：

```
lane i 有效  ⇔  (i + offset × 64 < ub)
```

含义："从第 offset×64 个元素开始，往后的 (ub − offset×64) 个元素里落在本帧内的部分"：

| 场景 | mask 形状 |
| --- | --- |
| 剩余量 ≥ 64（整帧需要） | 64 个 lane 全 1 |
| 剩余量 r ∈ (0, 64)（尾帧） | 前 r 个 lane 为 1，其余 0 |
| 剩余量 ≤ 0（帧已越过 x） | 64 个 lane 全 0（空转，vsts 不写任何元素） |

例（x = 150）：frame 0/1 全 1 → 元素 0..127 翻倍；frame 2 前 22 lane 为 1 → 元素 128..149
翻倍；frame 3 全 0 → 元素 192..255 保持 1。共 150 个 2。

## PLT vs PLTM（为什么需要两种指令）

- **PLT**：有状态。count 即输入也即输出，每次调用自动减一帧，适合 while/尾块（无法预知还有几帧）。
- **PLTM**：无状态。offset/ub 同时给出，帧号由循环变量直接提供，编译器可做循环展开、
  强度削减等优化；这也是编译器 `PLT→PLTM` 优化 pass（把固定步长循环里的 plt 改写成
  pltm(循环变量, 上界)）的目标形态。
- 两指令等价性：对第 f 帧，`pltm_b32(f, x)` ≡ 循环里第 f 次 `plt_b32(x − f×64, POST_UPDATE)`。
  因此本样例与 `plt_demo` 的输出逐元素一致，可互相印证。

## 为什么 load 不带 mask

同 `plt_demo`：V300 对齐 load（`vlds`）不带 predicate，整帧读入；mask 作用于
`vmuls`（active lane ×2）与 `vsts`（只写 active lane）。输出底色 1 由 host 预置在 GM。
