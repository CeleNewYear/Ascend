#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
plt/pltm 演示样例的数据生成脚本。

生成：
  input/input_x.bin : TOTAL_LENGTH 个 float32，全部为 1.0
  input/x.bin       : 1 个 uint32，运行时动态翻倍个数 x（默认随机；可用 --x 指定）
  output/golden.bin : TOTAL_LENGTH 个 float32，前 x 个为 2.0，其余为 1.0

用法：
  python3 scripts/gen_data.py            # x 随机（每次运行都不同，用于验证"运行时动态"）
  python3 scripts/gen_data.py --x 150    # 指定 x = 150
"""

import argparse
import os
import numpy as np

# 与 .asc 中的 TOTAL_LENGTH 保持一致
TOTAL_LENGTH = 256


def gen_golden_data_simple(x_count):
    # 输入：全 1 向量
    x = np.ones([1, TOTAL_LENGTH], dtype=np.float32)
    # 运行时翻倍个数 x（0 <= x <= TOTAL_LENGTH，uint32 存储）
    if x_count is None:
        x_count = int(np.random.randint(0, TOTAL_LENGTH + 1))  # [0, 256]，演示不同尾块/空帧
    # golden：前 x 个 2.0，其余 1.0
    golden = np.ones([1, TOTAL_LENGTH], dtype=np.float32)
    golden[0, :x_count] = 2.0

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x.tofile("./input/input_x.bin")
    np.array([x_count], dtype=np.uint32).tofile("./input/x.bin")
    golden.tofile("./output/golden.bin")
    print("gen done: TOTAL_LENGTH =", TOTAL_LENGTH, ", x =", x_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=None, help="fixed runtime active count x")
    args = parser.parse_args()
    gen_golden_data_simple(args.x)
