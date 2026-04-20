#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于Fluent连接等
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2026/1/25 22:23
version: V1.0
"""

import pandas as pd
import numpy as np
import os

# === 配置 ===
FLUENT_FILE = 'grotto_pure_volume.csv'  #  Fluent 导出原始文件 final_volume/
SENSOR_FILE = './data/_sensor_coords.csv'  # 训练用的传感器坐标
OUTPUT_FILE = './data/grotto_shape_points.csv'  # 输出结果


def fix_data():
    print(f"1. 读取 Fluent 文件: {FLUENT_FILE} ...")
    if not os.path.exists(FLUENT_FILE):
        print("错误：找不到 Fluent 文件！")
        return

    # 尝试读取，跳过可能的非数据头
    try:
        # 先读取前5行看看长什么样
        with open(FLUENT_FILE, 'r') as f:
            head = [next(f) for _ in range(5)]
        print("\n--- 文件前 5 行原始内容 ---")
        for line in head:
            print(line.strip())
        print("---------------------------\n")

        # 尝试自动读取
        try:
            df = pd.read_csv(FLUENT_FILE)
            if len(df.columns) < 2: df = pd.read_csv(FLUENT_FILE, sep=r'\s+')
        except:
            df = pd.read_csv(FLUENT_FILE, skiprows=1)  # 跳过第一行再试

    except Exception as e:
        print(f"读取失败: {e}")
        return

    print("检测到的列名:", list(df.columns))

    # === 步骤 A: 解决 X=0 问题 (交互式选列) ===
    print("\n>>> 请人工确认坐标列 (输入列名的序号, 从0开始) <<<")
    for i, col in enumerate(df.columns):
        # 显示每一列的数据范围，帮助判断
        col_data = pd.to_numeric(df[col], errors='coerce')
        vmin, vmax = col_data.min(), col_data.max()
        print(f"  [{i}] {col} \t(范围: {vmin:.2f} ~ {vmax:.2f})")

    # 让用户输入正确的列索引
    try:
        idx_x = int(input("请输入 X 坐标的列序号 (找范围是 -6~6 左右的, 不要选 0~0 的): "))
        idx_y = int(input("请输入 Y 坐标的列序号: "))
        idx_z = int(input("请输入 Z 坐标的列序号: "))
    except:
        print("输入错误，请输入数字！")
        return

    # 提取数据
    new_df = pd.DataFrame()
    new_df['x'] = pd.to_numeric(df.iloc[:, idx_x], errors='coerce')
    new_df['y'] = pd.to_numeric(df.iloc[:, idx_y], errors='coerce')
    new_df['z'] = pd.to_numeric(df.iloc[:, idx_z], errors='coerce')

    # 去除空值
    new_df = new_df.dropna()

    # === 步骤 B: 全局双窟对齐 ===
    if (new_df.max().max() - new_df.min().min()) > 100:
        new_df[['x', 'y', 'z']] /= 1000.0

    sensor_df = pd.read_csv(SENSOR_FILE)
    valid_sensors_all = [f"A{str(i).zfill(2)}" for i in range(1, 69)]
    sensor_df_all = sensor_df[sensor_df['sensor_id'].isin(valid_sensors_all)]

    ref_center = np.array([
        (sensor_df_all['x'].max() + sensor_df_all['x'].min()) / 2,
        (sensor_df_all['y'].max() + sensor_df_all['y'].min()) / 2,
        (sensor_df_all['z'].max() + sensor_df_all['z'].min()) / 2
    ])
    shape_center = np.array([
        (new_df['x'].max() + new_df['x'].min()) / 2,
        (new_df['y'].max() + new_df['y'].min()) / 2,
        (new_df['z'].max() + new_df['z'].min()) / 2
    ])

    # 3. 计算偏移量并只平移 X 和 Y
    offset_x = ref_center[0] - shape_center[0]
    offset_y = ref_center[1] - shape_center[1]
    # 🌟 放弃 Z 轴的中心对齐，保持 Fluent 原始的 Z 高度！

    print(f"应用平移修正: X+{offset_x:.2f}, Y+{offset_y:.2f}, Z 轴保持原高度不变！")

    new_df['x'] += offset_x
    new_df['y'] += offset_y
    # new_df['z'] += offset[2]  <--- 这行一定要删掉或注释掉！！！

    # offset = ref_center - shape_center
    # new_df['x'] += offset[0]
    # new_df['y'] += offset[1]
    # new_df['z'] += offset[2]

    # # === 步骤 C: 降采样与保存 ===
    # if len(new_df) > 200000:
    #     print(f"点数较多 ({len(new_df)})，降采样至 50000...")
    #     new_df = new_df.sample(n=200000, random_state=42)

    new_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ 修复完成！已保存至 {OUTPUT_FILE}")
    print("现在请重新运行 predict_model.py 查看结果。")


if __name__ == '__main__':
    fix_data()
