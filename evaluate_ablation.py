#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 绘制 Z 轴对比时序图。
特性：Times New Roman 字体，图例置底且位于左下角，增强 X 轴标注。
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 导入自定义模块
from predict_model import rebuild_scaler, load_merged_dataframe
from LogicAlgorithm.DL_method.GWNPINN.model_1 import SpatioTemporalPINN


def plot_z_timeseries_final():
    # ================= 1. 配置 =================
    TARGET_GROTTO_ID = 10
    TARGET_SENSOR = 'A24'
    PREDICT_STEPS = 864
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 文件夹映射（已根据反馈修正）
    MODELS_CFG = {
        'Z=1': './results_slices_FULL_Z_1',
        'Z=5': './results_slices_FULL_Z_5',
        'Z=10': './results_slices_FULL_Z_10',
        'Z=15': './results_slices_FULL_Z_15'
    }
    DATA_PATH = './data'

    # ================= 2. 字体与画布设置 =================
    # 🔥 核心修改：统一 Times New Roman 字体，调大字号
    plt.rcParams.update({
        'font.size': 15,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False
    })

    fig, ax = plt.subplots(figsize=(12, 6.5))

    print(f"Inference for Sensor {TARGET_SENSOR}...")
    first_ckpt = torch.load(os.path.join(MODELS_CFG['Z=10'], f'grotto_{TARGET_GROTTO_ID}_model_checkpoint.pth'),
                            map_location=DEVICE)
    sensor_ids = first_ckpt['graph_data']['sensor_ids']
    full_df = load_merged_dataframe(DATA_PATH, sensor_ids)
    target_idx = sensor_ids.index(TARGET_SENSOR)

    total_len = len(full_df)
    test_start_idx = int(total_len * 0.9)
    timestamps = full_df.index[test_start_idx: test_start_idx + PREDICT_STEPS]
    true_vals = full_df.iloc[test_start_idx: test_start_idx + PREDICT_STEPS, target_idx].values

    # ================= 3. 绘图：模型曲线 =================
    colors = {'Z=1': '#d62728', 'Z=5': '#ff7f0e', 'Z=10': '#1f77b4', 'Z=15': '#2ca02c'}
    linestyles = {'Z=1': '--', 'Z=5': '-.', 'Z=10': '-', 'Z=15': ':'}

    # 为了让 Observations 在图例最后，先循环画模型
    for mode, path in MODELS_CFG.items():
        ckpt_path = os.path.join(path, f'grotto_{TARGET_GROTTO_ID}_model_checkpoint.pth')
        if not os.path.exists(ckpt_path): continue

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

        # 数据处理
        temp_scaler = rebuild_scaler(checkpoint['scalers_params']['temp_scaler'])
        temp_min, temp_max = checkpoint['scalers_params']['global_min'], checkpoint['scalers_params']['global_max']
        full_df_norm = pd.DataFrame(temp_scaler.transform(full_df.values), index=full_df.index, columns=full_df.columns)

        adj_matrix = np.array(checkpoint['graph_data']['adj_matrix'])
        pre_adj_t = torch.tensor(adj_matrix, dtype=torch.float32, device=DEVICE)
        zone_ids = np.array(checkpoint['graph_data']['zone_ids'])
        zone_features = np.eye(3)[zone_ids].T
        seq_len = checkpoint['train_info']['seq_len']

        preds = []
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
                preds.append(pred_all[target_idx])

        preds = np.array(preds)
        z_order = 10 if mode == 'Z=10' else 5
        ax.plot(timestamps, preds, label=mode, color=colors[mode],
                linestyle=linestyles[mode], linewidth=2.5, zorder=z_order)

    # ================= 4. 绘图：真实值 (Observations) =================
    # 后画真实值，使其图例出现在最后，但通过 zorder=1 确保它在底层
    ax.plot(timestamps, true_vals, label='Observations',
            color='#7f7f7f', linewidth=5, alpha=0.35, zorder=1)

    # ================= 5. 修饰与保存 =================
    ax.set_ylabel('Temperature (°C)', fontweight='bold')
    # 🔥 核心修改：添加 X 轴标签
    ax.set_xlabel('Time', fontweight='bold')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=20)

    # 🔥 核心修改：图例位置设为左下角 (lower left)
    ax.legend(loc='lower left', frameon=True, fontsize=17)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('./Figure_8_TNR_Final.png', dpi=600)
    plt.show()


if __name__ == "__main__":
    plot_z_timeseries_final()