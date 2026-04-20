#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于合并侧视图和俯视图的效果（共用一根热力轴）
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2026/4/19 15:02
version: V1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= SCI 级画图全局配置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
# ====================================================

# ================= 🔧 终极微调控制台 (请在这里自由修改) =================
# 1. 调节两个子图之间的水平距离 (值越大，(a)和(b)离得越远，建议 0.2~0.35)
SUBPLOT_SPACING = 0.175

# 2. 调节左图(俯视图)在框内的偏移量 (单位: 米)
# 如果图偏右了想往左挪，请给 OFFSET_X_LEFT 设置正数 (例如 0.5 或 1.0)
# 如果想上下挪动，请修改 OFFSET_Y_LEFT
OFFSET_X_LEFT = 0.8  # ⬅️ 先默认帮设了 0.8，把图往左拉一点
OFFSET_Y_LEFT = 0.0

# 3. 如果想整体调节左框的胖瘦，可以修改这个比例 (左:右，目前是 1 : 1.5)
WIDTH_RATIOS = [1, 1.5]
# ======================================================================

DATA_DIR = "./results_slices_inference"
FILE_Z = os.path.join(DATA_DIR, "Slice_Z_1.0_20241006_0830.csv")  # 俯视图
FILE_X = os.path.join(DATA_DIR, "Slice_X_1.0_20241006_0830.csv")  # 侧视图
OUTPUT_FILE = os.path.join(DATA_DIR, "Combined_Slices_1.0_SCI_Final.png")


def main():
    print("正在加载 CSV 数据...")
    df_z = pd.read_csv(FILE_Z)
    df_x = pd.read_csv(FILE_X)

    global_min = min(df_z['temperature'].min(), df_x['temperature'].min())
    global_max = max(df_z['temperature'].max(), df_x['temperature'].max())
    print(f"共享热力轴范围: {global_min:.2f} °C ~ {global_max:.2f} °C")

    # 使用你的微调参数 WIDTH_RATIOS
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': WIDTH_RATIOS})

    # 🚀 使用你的微调参数 SUBPLOT_SPACING 来控制间距
    plt.subplots_adjust(left=0.05, right=0.9, bottom=0.15, top=0.95, wspace=SUBPLOT_SPACING)

    # ================= 绘制 (a) Z=1.0 俯视图 =================
    ax1 = axes[0]
    ax1.set_facecolor('#d3d3d3')

    grid_z = df_z.pivot(index='y', columns='x', values='temperature')
    X_z, Y_z = np.meshgrid(grid_z.columns, grid_z.index)

    cf1 = ax1.contourf(X_z, Y_z, grid_z.values, levels=100, cmap='rainbow',
                       vmin=global_min, vmax=global_max)

    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect('equal')

    # 🚀 使用微调参数 OFFSET_X_LEFT 和 OFFSET_Y_LEFT，让你随心所欲在框内挪动图像
    x1_min, x1_max = df_z['x'].min(), df_z['x'].max()
    margin_x1 = (x1_max - x1_min) * 0.05
    ax1.set_xlim(x1_min - margin_x1 + OFFSET_X_LEFT, x1_max + margin_x1 + OFFSET_X_LEFT)

    y1_min, y1_max = df_z['y'].min(), df_z['y'].max()
    margin_y1 = (y1_max - y1_min) * 0.05
    ax1.set_ylim(y1_min - margin_y1 + OFFSET_Y_LEFT, y1_max + margin_y1 + OFFSET_Y_LEFT)

    # ================= 绘制 (b) X=1.0 侧视图 =================
    ax2 = axes[1]
    ax2.set_facecolor('#d3d3d3')

    grid_x = df_x.pivot(index='z', columns='y', values='temperature')
    Y_x, Z_x = np.meshgrid(grid_x.columns, grid_x.index)

    cf2 = ax2.contourf(Y_x, Z_x, grid_x.values, levels=100, cmap='rainbow',
                       vmin=global_min, vmax=global_max)

    ax2.set_xlabel("y (m)")
    ax2.set_ylabel("z (m)")
    ax2.set_aspect('equal')

    # 右侧图保持强制对称留白
    y2_min, y2_max = df_x['y'].min(), df_x['y'].max()
    margin_y2 = (y2_max - y2_min) * 0.05
    ax2.set_xlim(y2_min - margin_y2, y2_max + margin_y2)

    z2_min, z2_max = df_x['z'].min(), df_x['z'].max()
    margin_z2 = (z2_max - z2_min) * 0.05
    ax2.set_ylim(z2_min - margin_z2, z2_max + margin_z2)

    # ================= 共享热力轴 (Colorbar) =================
    cbar = fig.colorbar(cf2, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Temperature (°C)', fontname='Times New Roman', fontsize=16)

    # ================= 绝对对齐的 (a) 和 (b) 标签 =================
    fig.canvas.draw()
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()

    fig.text(pos1.x0 + pos1.width / 2., 0.06, '(a)',
             fontsize=20, fontweight='bold', ha='center', va='center', fontname='Times New Roman')
    fig.text(pos2.x0 + pos2.width / 2., 0.06, '(b)',
             fontsize=20, fontweight='bold', ha='center', va='center', fontname='Times New Roman')

    # ================= 保存极高精度图像 =================
    plt.savefig(OUTPUT_FILE, dpi=1000, bbox_inches='tight')
    print(f"✅ 终极微调版拼图大功告成！已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()