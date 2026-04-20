#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于绘制多个传感器的温度时间序列对比图 (适用于 SCI 期刊)
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2026/4/20 14:33
version: V1.0
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import matplotlib.cm as cm
import numpy as np

# ================= 1. 🔧 SCI 绘图控制台 =================
# 🚀 核心控制开关：是否开启“突出核心矛盾”高亮模式？
ENABLE_HIGHLIGHT_MODE = True

# 1. 字体大小设置
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13  # 稍微调小一点图例字体，使其更好地适应左下角空间

# 全局字体强制为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# 2. 数据与路径配置
DATA_DIR = r"D:\PythonProject\MachineLearning\LogicAlgorithm\DL_method\GWNPINN\data"
OUTPUT_FILE = r"./Figure_Sensor_Temperature_Trends_SCI.png"

# 3. ⏱️ 时间截取范围
START_TIME_STR = "2024-10-06 00:00:00"
END_TIME_STR = "2024-10-09 12:00:00"

# 4. 🎯 传感器列表
TEMP_TABLE = ['a27_sensor', 'a28_sensor', 'a29_sensor', 'a30_sensor',
              'a31_sensor', 'a33_sensor', 'a35_sensor',  # a32_sensor脏数据删除
              'a36_sensor', 'a37_sensor', 'a38_sensor',
              'b31_sensor', 'b35_sensor', 'b36_sensor', 'b37_sensor',
              'b38_sensor',
              ]
# 自动过滤掉 A32 相关的脏数据
TEMP_TABLE = [sensor for sensor in TEMP_TABLE if not sensor.upper().startswith('A32')]


# ==============================================================


def main():
    start_time = pd.to_datetime(START_TIME_STR)
    end_time = pd.to_datetime(END_TIME_STR)

    fig, ax = plt.subplots(figsize=(16, 7))
    missing_files = []

    # 备用的彩色色卡
    colors = cm.tab20(np.linspace(0, 1, len(TEMP_TABLE)))

    print("正在加载数据并绘制曲线...")
    for idx, sensor in enumerate(TEMP_TABLE):
        prefix = sensor.split('_')[0].upper()
        filename = f"{prefix}.csv"
        filepath = os.path.join(DATA_DIR, filename)

        if not os.path.exists(filepath):
            missing_files.append(filename)
            continue

        try:
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])

            mask = (df['time'] >= start_time) & (df['time'] <= end_time)
            df_filtered = df.loc[mask]

            if not df_filtered.empty:
                # 🚀 触发器逻辑：核心主角高亮 vs 灰底对照
                if ENABLE_HIGHLIGHT_MODE:
                    if prefix == 'A33':
                        line_color = '#d62728'
                        line_width = 3.5
                        alpha_val = 1.0
                        z_order = 10
                    elif prefix == 'A37':
                        line_color = '#1f77b4'
                        line_width = 3.5
                        alpha_val = 1.0
                        z_order = 10
                    else:
                        line_color = 'gray'
                        line_width = 1.0
                        alpha_val = 0.3
                        z_order = 1
                else:
                    line_color = colors[idx]
                    line_width = 2.0
                    alpha_val = 0.85
                    z_order = 5

                # 画线
                ax.plot(df_filtered['time'], df_filtered['air_temperature'],
                        label=prefix, color=line_color, linewidth=line_width,
                        alpha=alpha_val, zorder=z_order)

        except Exception as e:
            print(f"❌ 处理文件 {filename} 时出错: {e}")

    # ================= 🚀 绘制冷锋区间阴影 =================
    if ENABLE_HIGHLIGHT_MODE:
        shade_start = pd.to_datetime("2024-10-08 00:00:00")
        shade_end = pd.to_datetime("2024-10-08 12:00:00")

        ax.axvspan(shade_start, shade_end, facecolor='#add8e6', alpha=0.3, zorder=0)

        shade_midpoint = shade_start + (shade_end - shade_start) / 2
        ax.text(shade_midpoint, 0.95, "Severe Cold Front",
                transform=ax.get_xaxis_transform(),
                color='#005b96', fontsize=15, fontweight='bold',
                ha='center', va='top', zorder=12)

    # ================= 图表美化与布局优化 =================
    ax.set_xlabel("Time", fontweight='bold')
    ax.set_ylabel("Temperature (°C)", fontweight='bold')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate(rotation=30)

    ax.grid(True, linestyle='--', alpha=0.6, color='gray', zorder=0)

    # 🚀 完美的内嵌图例排版：放置在图表内部的左下角
    # 设置列数为 6 列（假设有11-12个传感器，刚好排成两行完整的矩形块，完美契合红框）
    ax.legend(loc='lower left',
              ncol=6,
              title="Sensor ID", title_fontproperties={'weight': 'bold'},
              framealpha=0.9, edgecolor='black')

    # 注意：这里删除了 plt.subplots_adjust(bottom=0.25)
    # 因为图例现在移到了图表内部，不再需要强行扩大外部底边的留白了。

    plt.savefig(OUTPUT_FILE, dpi=1000, bbox_inches='tight')
    print(f"\n✅ 核心矛盾突出版 SCI 图 (左下角图例) 已生成: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()