#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于
上半部分画 1D 时序曲线，下半部分横向排布 4 张 2D 切片，共享同一个全局热力轴 Figure 5
 5author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2026/4/19 18:56
version: V1.0
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime
import os

# ================= 🔧 SCI 级微调控制台 =================
# 1. 字体大小设置
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# 2. 布局间距调节
SUBPLOT_HSPACE = 0.3  # 上图(a)与下方四张图的上下距离
SUBPLOT_WSPACE = 0.05   # 四张子图之间的水平间距
LABEL_Y_OFFSET = -0.15  # (b)(c)(d)(e) 标签与图框的上下距离

# 3. 调节图框内热力图的左右偏移量 (单位: 米，用来调整石窟在灰色框里的位置)
PANEL_OFFSET_X = 1.0

# 4. 子图标签字号与样式
LABEL_FONT_SIZE = 25
LABEL_FONT_WEIGHT = 'bold'

# 5. 🚀 绝杀：调节整个下半部分 (四个切片 + 色带) 的全局水平移动！
# 负值 (-0.02, -0.05 等) 代表整体向左平移，用来完美对齐上方的图 (a)！
BOTTOM_ROW_OFFSET_X = -0.05
# ====================================================

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

DATA_DIR = "./results_slices_inference"
ENV_DATA_FILE = "./data/_core_area_5min.csv"

FILES = [
    os.path.join(DATA_DIR, "Slice_Z_1.0_20241008_0600.csv"),
    os.path.join(DATA_DIR, "Slice_Z_1.0_20241008_1200.csv"),
    os.path.join(DATA_DIR, "Slice_Z_1.0_20241008_1800.csv"),
    os.path.join(DATA_DIR, "Slice_Z_1.0_20241008_2355.csv")
]

SUB_LABELS = ["(b)", "(c)", "(d)", "(e)"]
OUTPUT_FILE = os.path.join(DATA_DIR, "Figure_5_Spatiotemporal_Minimalist.png")


def main():
    print("正在加载数据并准备高清绘图...")

    if not os.path.exists(ENV_DATA_FILE):
        print(f"❌ 错误：找不到外部环境数据文件 {ENV_DATA_FILE}！")
        return

    df_env = pd.read_csv(ENV_DATA_FILE)
    df_env['time'] = pd.to_datetime(df_env['time'])
    target_date = pd.to_datetime("2024-10-08").date()
    df_day = df_env[df_env['time'].dt.date == target_date].copy()

    if df_day.empty:
        print("❌ 错误：在 CSV 中没有找到 2024-10-08 的数据！")
        return
    df_day.sort_values('time', inplace=True)

    dfs = [pd.read_csv(f) for f in FILES]

    # 解绑温标
    env_min = df_day['air_temperature'].min()
    env_max = df_day['air_temperature'].max()
    EXT_Y_MIN = np.floor(env_min) - 1.0
    EXT_Y_MAX = np.ceil(env_max) + 1.0

    slice_min = min([df['temperature'].min() for df in dfs])
    slice_max = max([df['temperature'].max() for df in dfs])
    INT_CBAR_MIN = np.floor(slice_min * 10) / 10.0
    INT_CBAR_MAX = np.ceil(slice_max * 10) / 10.0

    contour_levels = np.linspace(INT_CBAR_MIN, INT_CBAR_MAX, 100)

    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 2.8], hspace=SUBPLOT_HSPACE, wspace=SUBPLOT_WSPACE)

    # ================= (a) 1D 真实外部曲线部分 =================
    ax_curve = fig.add_subplot(gs[0, :])
    ax_curve.plot(df_day['time'], df_day['air_temperature'], color='black', linewidth=2.5)

    target_times = [
        datetime(2024, 10, 8, 6, 0), datetime(2024, 10, 8, 12, 0),
        datetime(2024, 10, 8, 18, 0), datetime(2024, 10, 8, 23, 55)
    ]

    actual_plot_times = []
    target_temps = []
    for t in target_times:
        idx = (df_day['time'] - t).abs().idxmin()
        actual_plot_times.append(df_day.loc[idx, 'time'])
        target_temps.append(df_day.loc[idx, 'air_temperature'])

    ax_curve.scatter(actual_plot_times, target_temps, color='red', s=120, zorder=5)

    # 🚀 绝杀 1：针对最后一个点 (e)，改变对齐方式并向左偏移，防止出界！
    for i, (t, temp) in enumerate(zip(actual_plot_times, target_temps)):
        ha_align = 'right' if i == 3 else 'center'  # 最后一个点靠右对齐(向左延伸)
        offset_x = -5 if i == 3 else 0  # 最后一个点额外向左平移 5 像素

        ax_curve.annotate(SUB_LABELS[i], (t, temp), textcoords="offset points", xytext=(offset_x, 12),
                          ha=ha_align, fontsize=18, color='red', fontweight='bold')

    ax_curve.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_curve.set_xlim(df_day['time'].iloc[0], df_day['time'].iloc[-1])
    ax_curve.set_ylim(EXT_Y_MIN, EXT_Y_MAX)
    ax_curve.set_ylabel("Temperature (°C)")
    # ax_curve.set_title("(a) External Temperature Forcing (Measured at Entrance)", fontweight='bold', pad=20)

    ax_curve.text(0.5, -0.25, '(a)', transform=ax_curve.transAxes,
                  fontsize=LABEL_FONT_SIZE, fontweight=LABEL_FONT_WEIGHT, ha='center', va='top')

    # ================= (b)-(e) 2D 切片群部分 =================
    axes_slices = []
    cf = None

    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        axes_slices.append(ax)
        df = dfs[i]
        ax.set_facecolor('#d3d3d3')

        grid = df.pivot(index='y', columns='x', values='temperature')
        X, Y = np.meshgrid(grid.columns, grid.index)

        cf = ax.contourf(X, Y, grid.values, levels=contour_levels, cmap='rainbow')

        ax.set_aspect('equal')
        ax.set_xlabel("X (m)", labelpad=10)

        # 🚀 绝杀 2：取消隐藏条件，让 4 张图全部显示 Y 轴和刻度！
        ax.set_ylabel("Y (m)")

        x_min, x_max = df['x'].min(), df['x'].max()
        margin_x = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - margin_x + PANEL_OFFSET_X, x_max + margin_x + PANEL_OFFSET_X)

        y_min, y_max = df['y'].min(), df['y'].max()
        margin_y = (y_max - y_min) * 0.05
        ax.set_ylim(y_min - margin_y, y_max + margin_y)

        ax.text(0.5, LABEL_Y_OFFSET, SUB_LABELS[i], transform=ax.transAxes,
                fontsize=LABEL_FONT_SIZE, fontweight=LABEL_FONT_WEIGHT, ha='center', va='top')

    # ================= 统一的紧凑矩形 Colorbar =================
    cbar = fig.colorbar(cf, ax=axes_slices, orientation='vertical', fraction=0.015, pad=0.03)
    cbar.set_label('Temperature (°C)', fontname='Times New Roman', fontsize=20)

    # ================= 整体平移下半部分框架 =================
    if BOTTOM_ROW_OFFSET_X != 0:
        fig.canvas.draw()
        for ax in axes_slices:
            pos = ax.get_position()
            ax.set_position([pos.x0 + BOTTOM_ROW_OFFSET_X, pos.y0, pos.width, pos.height])
        cbar_pos = cbar.ax.get_position()
        cbar.ax.set_position([cbar_pos.x0 + BOTTOM_ROW_OFFSET_X, cbar_pos.y0, cbar_pos.width, cbar_pos.height])

    plt.savefig(OUTPUT_FILE, dpi=1000, bbox_inches='tight')
    print(f"✅ (e)标签防碰撞、全 Y 轴展示版组合图已生成: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()