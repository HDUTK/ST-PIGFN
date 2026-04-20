#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于Fluent文件
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2026/1/25 22:34
version: V1.0
"""

import pandas as pd
import numpy as np

# === 配置 ===
FLUENT_FILE = 'fluent_clip_export'  # 原始 Fluent 文件
OUTPUT_FILE = './data/grotto_shape_points.csv'


def extract_pure():
    print(f"正在读取 Fluent 文件: {FLUENT_FILE} ...")

    # 1. 智能读取（跳过非数据行）
    try:
        # 尝试读取前几行判断跳过多少行，通常 Fluent ASCII 第一行是 info
        df = pd.read_csv(FLUENT_FILE, skiprows=1)
        # 如果列数太少，可能是空格分隔
        if len(df.columns) < 2:
            df = pd.read_csv(FLUENT_FILE, skiprows=1, sep=r'\s+')
    except:
        print("读取失败，请检查文件格式。")
        return

    print("原始列名:", list(df.columns))

    # 2. 根据之前的反馈，锁定正确的列

    new_df = pd.DataFrame()
    new_df['x'] = df.iloc[:, 1]  # 第2列
    new_df['y'] = df.iloc[:, 2]  # 第3列
    new_df['z'] = df.iloc[:, 3]  # 第4列

    # 3. 去除空值
    new_df = new_df.dropna()

    # 4. 检查单位 (修正毫米 -> 米)
    # 如果范围 > 100，说明是 mm，除以 1000
    if (new_df.max().max() - new_df.min().min()) > 100:
        print("检测到单位为 mm，转换为 m...")
        new_df = new_df / 1000.0
    else:
        print("检测到单位为 m，保持不变。")

    # 5. 【关键】采样更多点！
    # 之前 5万点太少，导致出现“空洞”。现在增加到 20万点。
    # 既然只是用来做 Mask，20万点计算也很快。
    # if len(new_df) > 200000:
    #     print(f"采样点数从 {len(new_df)} 降至 200000 (提高密度以消除空洞)...")
    #     new_df = new_df.sample(n=200000, random_state=42)
    # 为了画出丝滑的边界，需要“百万级”的点云
    # if len(new_df) > 1000000:
    #     print(f"采样点数从 {len(new_df)} 降至 1000000 (百万级密度)...")
    #     new_df = new_df.sample(n=1000000, random_state=42)
    # else:
    #     print(f"保留全量点数: {len(new_df)}")

    print(f"使用全量 Fluent 点云数据，共 {len(new_df)} 个点。")
    print("这能确保边界最精确，但初始化Masker可能会慢几十秒，请耐心等待。")

    # 6. 保存 (绝对不进行中心平移！)
    new_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 纯净数据已提取至 {OUTPUT_FILE}")
    print("坐标未做平移，保持原始位置。")


if __name__ == '__main__':
    extract_pure()
