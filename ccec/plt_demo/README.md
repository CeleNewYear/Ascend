# plt_demo

Ascend950（DAV_3510）**PLT** 指令验证与演示样例：用 ccec 内建 `plt_b32(count, POST_UPDATE)`
生成运行时 predicate mask，做一次 predicated 翻倍，验证输出为"前 x 个 2、其余 1"。
**x 是运行时动态值**（host 写入 `input/x.bin`，kernel 从 GM 解引用读取），不是编译期常量。

## 文件结构

| 文件 | 说明 |
| --- | --- |
| `plt_demo.asc` | kernel（`plt_b32` + `vlds`/`vmuls`/`vsts`）与 host 入口，ASC 单文件工程 |
| `data_utils.h` | 输入/输出文件读写辅助工具 |
| `CMakeLists.txt` | ASC CMake 构建脚本，目标 `dav-3510` |
| `scripts/gen_data.py` | 生成全 1 输入、运行时 x、golden（前 x 个 2） |

## 构建与运行

```bash
# 1. 生成数据（默认 x 随机；固定 x：python3 scripts/gen_data.py --x 150）
python3 scripts/gen_data.py

# 2. 构建（真实 NPU 用 npu；本机无 NPU 时用 sim 模拟器）
cmake -B build -DCMAKE_ASC_RUN_MODE=npu  -DCMAKE_ASC_ARCHITECTURES=dav-3510   # 或
cmake -B build -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build

# 3. 运行（在 demo 目录下，输入输出为相对路径；可选参数为 device id，默认 0）
./build/demo
```

预期输出：`runtime x (active_count) = <x>`；`test pass!`（256 个元素全部核对通过：
前 x 个 == 2.0，其余 == 1.0，同时与 `golden.bin` 逐元素一致）。
本样例已在 dav-3510 模拟器（sim 模式）实测：x=150 与 x=77 均 `test pass!`。

> host→device 数据拷贝请使用 `ACL_MEMCPY_HOST_TO_DEVICE`（与 `cycle_count_demo` 一致）；
> 模拟器环境下 `ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE`（reduce_demo 真机写法）不生效，输出会全 0。

## 验证内容

输入是 256 个 `1.0f`（b32 一帧 64 lane，共 4 帧）。把 `x.bin` 里的值改成任意 `[0, 256]`
并重跑（`gen_data.py` 会同步更新 golden），输出始终是前 x 个 2、其余 1 —— 这直接证明
mask 是**运行时**生成的，而不是编译期常量折叠。

覆盖建议：x = 0（mask 全 0，全输出 1）、x = 64（恰好整帧）、x = 150（整帧 + 尾帧）、
x = 256（全量 4 帧）。

## PLT 指令语义讲解

- 一帧 = VL = 256B；float32 一帧 64 lane。predicate（`vector_bool`）256 bit，bit i 对应 lane i。
- `plt_b32(count, POST_UPDATE)`：
  - 返回 mask：前 `min(count, 64)` 个 lane 为 1，其余为 0；
  - 副作用：`count` 被自动扣减一帧（`count -= 64`），即 **POST_UPDATE**。
- PLT 因此是**有状态、计数递减**式：同一个标量既是输入（还剩多少元素）也是输出（还剩多少没处理）。
  天然匹配 while 风格尾块循环。
- **实现结构说明**：本样例实测发现 bisheng（dav-c310-vec）后端对 `__simd_vf__` 函数内的
  while/if 条件控制流 + 向量指令组合会段错误崩溃；因此分支（整帧/尾帧判断）全部放在
  kernel 标量上下文（`num_full = x/64`、`tail = x%64`），两个 vf 函数内部只保留 for 直落循环：
  - `plt_full_frames_vf`：连续 `plt_b32(remain, POST_UPDATE)`，remain 每帧自动 -64
    （演示 POST_UPDATE 计数递减）；循环结束 remain 恰好 == tail；
  - `plt_tail_frame_vf`：尾帧调用一次 `plt_b32(tail, ...)`，生成前 tail 个 lane 的 mask
    （此时 remain_t 下溢无妨，因为不再使用）。
- 对照工程 `../pltm_demo`：同一验证用 `pltm_b32(offset, ub)` 无状态写法实现，结果必须一致。

## 验证结果（本机 dav-3510 模拟器实测）

- x = 150：256/256 元素核对通过，`test pass!`
- x = 77：256/256 元素核对通过，`test pass!`（可看到第 2 帧仅前 13 lane 为 2，
  即第 77 个元素起恢复为 1，尾帧 mask 语义正确）
- 换 x 只需 `python3 scripts/gen_data.py --x <新值>` 后重跑，golden 同步更新。

## 为什么 load 不带 mask

V300 普通对齐 load（`vlds`）没有 mask 输入（指令编码不含 predicate），总是整帧读入；
真正受 mask 控制的是计算（`vmuls` 的 `.m/.x` 变体）与写出（`vsts` predicated store）。
本样例用 `vmuls(..., MODE_ZEROING)` + masked `vsts`：active lane 写 2，其余 lane 保持
GM 输出底色 1（host 在启动 kernel 前把输出缓冲预置为全 1）。
