#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于数据转换 (ASCII -> CSV)
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2026/1/25 17:13
version: V1.0
"""

# convert_ascii.py
import pandas as pd
import os
import matplotlib.pyplot as plt


# === 配置 ===
# 1. Fluent 导出文件名 (包含路径)
# close_dinuan_wentai_8_3_9_40/close_dinuan_wentai_8_3_9_41/close_dinuan_wentai_8_3_9_42
INPUT_FILE = 'close_dinuan_wentai_8_3_9_42'
# 2. 输出的标准 CSV 文件名
OUTPUT_FILE = './data/grotto_shape_points.csv'


def convert_fluent_ascii():
    print(f"正在读取文件: {INPUT_FILE} ...")

    if not os.path.exists(INPUT_FILE):
        print("错误：找不到文件，请确认文件名和路径是否正确。")
        return

    # 尝试读取。Fluent ASCII 可能是逗号分隔，也可能是空格分隔。
    # 我们先尝试读取前几行看看格式
    try:
        # 尝试作为 CSV 读取，跳过可能的非数据头（如果有 metadata）
        # header=0 表示第一行是列名
        df = pd.read_csv(INPUT_FILE)

        # 如果读出来只有1列，说明可能是空格分隔
        if len(df.columns) < 2:
            df = pd.read_csv(INPUT_FILE, sep=r'\s+')

    except Exception as e:
        print(f"读取失败，尝试跳过第一行重试... 错误: {e}")
        try:
            # 有时候第一行是类似 "Node" 这样的说明，第二行才是列名
            df = pd.read_csv(INPUT_FILE, skiprows=1)
        except:
            print("无法解析文件格式，请打开文件截图给我看前几行内容。")
            return

    print("原始列名:", df.columns.tolist())

    # === 自动寻找坐标列 ===
    # Fluent 导出的列名千奇百怪，模糊匹配
    col_map = {}
    for col in df.columns:
        c_lower = str(col).lower()
        if 'x' in c_lower and ('node' in c_lower or 'coor' in c_lower):
            col_map['x'] = col
        elif 'y' in c_lower and ('node' in c_lower or 'coor' in c_lower):
            col_map['y'] = col
        elif 'z' in c_lower and ('node' in c_lower or 'coor' in c_lower):
            col_map['z'] = col

    # 如果没找到标准列名，可能就是第1,2,3列（假设文件里全是数据）
    if len(col_map) < 3:
        print("警告：无法通过列名识别坐标，尝试直接使用第1、2、3列...")
        col_list = df.columns.tolist()
        col_map['x'] = col_list[0]  # 这里假设顺序是 noder, x, y, z，通常第0列是ID
        # 如果第0列看起来像整数索引（1,2,3...），则取 1,2,3 列作为坐标
        if df.iloc[0, 0] == 1 and df.iloc[1, 0] == 2:
            col_map['x'] = col_list[1]
            col_map['y'] = col_list[2]
            col_map['z'] = col_list[3]
        else:
            col_map['x'] = col_list[0]
            col_map['y'] = col_list[1]
            col_map['z'] = col_list[2]

    print(f"使用列: X={col_map['x']}, Y={col_map['y']}, Z={col_map['z']}")

    # 提取数据
    new_df = pd.DataFrame()
    new_df['x'] = df[col_map['x']]
    new_df['y'] = df[col_map['y']]
    new_df['z'] = df[col_map['z']]

    # === 单位检查 ===
    # Fluent 默认可能是米，也可能是毫米。根据模型大小判断。
    # 比如：如果 x 坐标出现了 5000，那肯定是毫米；如果是 5.0，那是米。
    max_val = new_df.max().max()
    if max_val > 1000:
        print(f"检测到数值较大 ({max_val})，推测单位为毫米，正在转换为米...")
        new_df = new_df / 1000.0

    # 降采样 (如果是全量网格，点太多会拖慢画图速度，保留 5万个点足够定义形状了)
    if len(new_df) > 200000:
        print(f"点数较多 ({len(new_df)})，降采样至 50000 点...")
        new_df = new_df.sample(n=200000, random_state=42)

    new_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 转换成功！已保存至 {OUTPUT_FILE}")


if __name__ == '__main__':
    convert_fluent_ascii()

    df = pd.read_csv('./data/grotto_shape_points.csv')
    # 只画前10000个点看看形态
    sample = df.sample(10000)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(sample['x'], sample['y'], sample['z'], s=0.1)
    plt.show()