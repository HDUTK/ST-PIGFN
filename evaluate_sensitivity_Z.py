#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 绘制 Z 轴敏感性分析柱状图，统一字体为 Times New Roman，修复排版重叠
author: TangKan
contact: 785455964@qq.com
time: 2026/4/15
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 导入预测与模型相关模块
from predict_model import rebuild_scaler, load_merged_dataframe
from LogicAlgorithm.DL_method.GWNPINN.model_1 import SpatioTemporalPINN


def plot_z_sensitivity_grouped_bar_final():
    # ================= 1. 环境配置 =================
    TARGET_GROTTO_ID = 10
    PREDICT_STEPS = 864
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Z_VALUES = [1, 5, 10, 15]
    MODELS_DIRS = {z: f'./results_slices_FULL_Z_{z}' for z in Z_VALUES}
    DATA_PATH = './data'

    # ================= 2. Times New Roman 字体设置 =================
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 18,
        'axes.unicode_minus': False,
        'mathtext.fontset': 'stix'
    })

    print("Loading test data and models...")
    first_ckpt = torch.load(os.path.join(MODELS_DIRS[1], f'grotto_{TARGET_GROTTO_ID}_model_checkpoint.pth'),
                            map_location=DEVICE)
    sensor_ids = first_ckpt['graph_data']['sensor_ids']
    full_df = load_merged_dataframe(DATA_PATH, sensor_ids)

    total_len = len(full_df)
    test_start_idx = int(total_len * 0.9)
    true_vals_global = full_df.iloc[test_start_idx: test_start_idx + PREDICT_STEPS].values

    results = []

    # ================= 3. 循环计算不同 Z 值的模型指标 =================
    for z in Z_VALUES:
        print(f"Evaluating Z = {z} ...")
        path = MODELS_DIRS[z]
        ckpt_path = os.path.join(path, f'grotto_{TARGET_GROTTO_ID}_model_checkpoint.pth')

        if not os.path.exists(ckpt_path):
            continue

        checkpoint = torch.load(ckpt_path, map_location=DEVICE)

        model = SpatioTemporalPINN(
            num_nodes=checkpoint['config']['num_nodes'],
            sensor_coords_norm=checkpoint['config']['sensor_coords_norm'],
            in_dim=checkpoint['config']['input_dim'],
            out_dim=checkpoint['config']['out_dim'],
            use_q_net=True
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()

        temp_scaler = rebuild_scaler(checkpoint['scalers_params']['temp_scaler'])
        temp_min, temp_max = checkpoint['scalers_params']['global_min'], checkpoint['scalers_params']['global_max']
        full_df_norm = pd.DataFrame(temp_scaler.transform(full_df.values), index=full_df.index, columns=full_df.columns)

        adj_matrix = np.array(checkpoint['graph_data']['adj_matrix'])
        pre_adj_t = torch.tensor(adj_matrix, dtype=torch.float32, device=DEVICE)
        zone_ids = np.array(checkpoint['graph_data']['zone_ids'])
        zone_features = np.eye(3)[zone_ids].T
        seq_len = checkpoint['train_info']['seq_len']

        preds_global = []
        with torch.no_grad():
            for i in range(test_start_idx, test_start_idx + PREDICT_STEPS):
                hist_data = full_df_norm.values[i - seq_len: i].T
                feat_temp = hist_data[np.newaxis, :, :]
                feat_zone = np.tile(zone_features[:, :, np.newaxis], (1, 1, seq_len))
                x_input = np.concatenate([feat_temp, feat_zone], axis=0)
                x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                _, sensor_feats = model.gwn(x_tensor, pre_adj_t)
                batch_sensor_coords = model.sensor_coords.unsqueeze(0)
                batch_t_zero = torch.zeros((1, len(sensor_ids), 1), device=DEVICE)
                interp_feats = model.interpolator(batch_sensor_coords, model.sensor_coords, sensor_feats)
                decoder_input = torch.cat([batch_sensor_coords, batch_t_zero, interp_feats], dim=-1)

                y_hybrid_norm = model.decoder(decoder_input).squeeze(-1)
                pred_all = y_hybrid_norm.cpu().numpy().flatten() * (temp_max - temp_min) + temp_min
                preds_global.append(pred_all)

        preds_global = np.array(preds_global)
        mask = ~np.isnan(true_vals_global) & ~np.isnan(preds_global)

        rmse = np.sqrt(mean_squared_error(true_vals_global[mask], preds_global[mask]))
        mae = mean_absolute_error(true_vals_global[mask], preds_global[mask])
        results.append({'Z': z, 'RMSE': rmse, 'MAE': mae})

    # ================= 4. 绘图逻辑 =================
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(9, 6))

    x_labels = [f"Z={int(z)}\n(Isotropic)" if z == 1 else f"Z={int(z)}" for z in df['Z']]
    x = np.arange(len(x_labels))
    width = 0.35

    colors = ['#d62728' if z == 1 else '#1f77b4' for z in df['Z']]

    rects1 = ax.bar(x - width / 2, df['RMSE'], width, label='RMSE', color=colors, edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x + width / 2, df['MAE'], width, label='MAE', color=colors, edgecolor='black', hatch='//',
                    alpha=0.7)

    # 🔥 将 Y 轴上限乘数由 1.25 提高到 1.35，给图例和数字腾出足够的安全空间
    max_val = max(df['RMSE'].max(), df['MAE'].max())
    ax.set_ylim(0, max_val * 1.25)

    # 🔥 减小高度偏移量 (h + 0.0015)，让数字更贴近柱子顶部
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.0003, f'{h:.4f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=15)

    autolabel(rects1)
    autolabel(rects2)

    ax.set_ylabel('Error (°C)', fontweight='bold')
    ax.set_xlabel('Anisotropic Penalty Factor ($Z$)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    # 锁定坐标范围后绘制定向虚线
    xlim = ax.get_xlim()
    rmse_base = df[df['Z'] == 1]['RMSE'].values[0]
    mae_base = df[df['Z'] == 1]['MAE'].values[0]

    ax.hlines(y=rmse_base, xmin=0 - width / 2, xmax=xlim[1], color='#d62728', linestyle='--', alpha=0.4, zorder=0)
    ax.hlines(y=mae_base, xmin=0 + width / 2, xmax=xlim[1], color='#d62728', linestyle='--', alpha=0.4, zorder=0)
    ax.set_xlim(xlim)

    # 自定义图例配置
    import matplotlib.patches as mpatches
    rmse_p = mpatches.Patch(facecolor='gray', edgecolor='black', label='RMSE')
    mae_p = mpatches.Patch(facecolor='gray', edgecolor='black', hatch='//', alpha=0.7, label='MAE')

    # 图例依然放在左上角，但由于 Y 轴拉高，它会自动上升，不再遮挡文字
    ax.legend(handles=[rmse_p, mae_p], loc='upper left')

    ax.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('./Figure_7_Z_Sensitivity_TNR_Fixed.png', dpi=1000)
    plt.show()


if __name__ == "__main__":
    plot_z_sensitivity_grouped_bar_final()