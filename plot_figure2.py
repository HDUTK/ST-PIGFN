#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于生成 SCI 论文 Figure 3:
Panel A: 训练收敛曲线 (Loss components)
Panel B-E: 4个极具代表性的传感器在测试集上的连续时序预测对比 (体现3D空间异质性)
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2026/1/9 19:17
version: V1.0
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# 导入原有的库
from predict_model import rebuild_scaler, load_merged_dataframe
from LogicAlgorithm.DL_method.GWNPINN.model_1 import SpatioTemporalPINN


def plot_figure_3():
    # ================= 1. 🔧 字体与排版微调控制台 =================
    # 1. 字体大小设置
    AXES_LABEL_FONTSIZE = 16  # 横纵坐标轴标题字体大小 (例如: Time, Temperature, Epochs)
    PANEL_LABEL_FONTSIZE = 22.5  # (a), (b), (c), (d), (e) 标签的字体大小
    TICK_LABEL_FONTSIZE = 14  # 坐标轴刻度数字的字体大小 (如 10-06 12:00)
    LEGEND_FONTSIZE = 13  # 图例字体大小

    # 2. 标签位置微调
    LABEL_Y_OFFSET_A = -0.1575  # (a) 标签在垂直方向上的基准偏移量
    LABEL_Y_OFFSET_B_E = -0.425  # (b)-(e) 标签在垂直方向上的偏移量

    # 3. 🚀 绝杀：整体图(a)框上下移动控制 (解绑了标签，标签不会跟着动！)
    # 负数代表整个 Loss 图框向下移动，正数向上。可以自由调节它来拉近和字母(a)的距离
    PANEL_A_BOX_Y_OFFSET = -0.04
    # ==============================================================

    TARGET_GROTTO_ID = 10
    TARGET_SENSORS = ['D23', 'C09', 'B03', 'A35']
    PREDICT_STEPS = 864
    doc = 'results_slices_20260406_02_有Q_！'

    PANEL_TITLES = ['(a)', '(b)', '(c)', '(d)', '(e)']

    LOSS_CSV = './' + doc + f'/grotto_{TARGET_GROTTO_ID}_loss_history.csv'
    CHECKPOINT_PATH = './' + doc + f'/grotto_{TARGET_GROTTO_ID}_model_checkpoint.pth'
    DATA_PATH = './data'
    COORDS_PATH = './data/_sensor_coords.csv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================= 2. 准备画图环境与全局字体设置 =================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'

    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(bottom=0.18)
    gs = GridSpec(2, 3, width_ratios=[1.2, 1, 1], hspace=0.55, wspace=0.28)

    # ================= 3. 画 Panel A (Loss 曲线) =================
    ax1 = fig.add_subplot(gs[:, 0])
    if os.path.exists(LOSS_CSV):
        df_loss = pd.read_csv(LOSS_CSV)
        epochs = df_loss['epoch']

        ax1.plot(epochs, df_loss['total_loss'], label=r'Total Loss ($\mathcal{L}_{total}$)', color='black', linewidth=2)
        ax1.plot(epochs, df_loss['loss_recon'], label=r'Recon Anchor Loss ($\mathcal{L}_{recon}$)', color='#1f77b4',
                 linestyle='--')
        ax1.plot(epochs, df_loss['loss_data'], label=r'Forecast Loss ($\mathcal{L}_{data}$)', color='#ff7f0e',
                 linestyle='-.')
        ax1.plot(epochs, df_loss['loss_bc'], label=r'Boundary Loss ($\mathcal{L}_{BC}$)', color='#2ca02c',
                 linestyle=':')

        ax1.set_xlabel('Epochs', fontweight='bold', fontsize=AXES_LABEL_FONTSIZE)
        ax1.set_ylabel('Loss Value', fontweight='bold', fontsize=AXES_LABEL_FONTSIZE)

        ax1.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
        ax1.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
        ax1.grid(True, linestyle='--', alpha=0.6)
    else:
        print(f"Error: 找不到 Loss 文件 {LOSS_CSV}")
        return

    # ================= 4. 准备数据与模型推理 =================
    print("Loading Model for Time Series Prediction...")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    except:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    config = checkpoint['config']
    scalers_params = checkpoint['scalers_params']
    temp_scaler = rebuild_scaler(scalers_params['temp_scaler'])
    temp_min = scalers_params['global_min']
    temp_max = scalers_params['global_max']
    graph_data = checkpoint['graph_data']
    adj_matrix = np.array(graph_data['adj_matrix'])
    sensor_ids = graph_data['sensor_ids']
    train_info = checkpoint['train_info']
    seq_len = train_info['seq_len']

    target_indices = {}
    for s_id in TARGET_SENSORS:
        if s_id in sensor_ids:
            target_indices[s_id] = sensor_ids.index(s_id)
        else:
            target_indices[s_id] = None

    model = SpatioTemporalPINN(
        num_nodes=config['num_nodes'],
        sensor_coords_norm=config['sensor_coords_norm'],
        in_dim=config['input_dim'],
        out_dim=config['out_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    full_df = load_merged_dataframe(DATA_PATH, sensor_ids)
    full_df_norm = pd.DataFrame(temp_scaler.transform(full_df.values), index=full_df.index, columns=full_df.columns)

    coords_df = pd.read_csv(COORDS_PATH)
    coords_df = coords_df[coords_df['grottoe_id'] == TARGET_GROTTO_ID]
    coords_df = coords_df.set_index('sensor_id').loc[sensor_ids].reset_index()
    zone_ids = coords_df['zone_id'].values.astype(int)
    zone_features = np.eye(3)[zone_ids].T

    total_len = len(full_df)
    test_start_idx = int(total_len * 0.9)
    data_values_norm = full_df_norm.values
    data_values_raw = full_df.values
    pre_adj_t = torch.tensor(adj_matrix, dtype=torch.float32, device=device)

    preds_dict = {s: [] for s in TARGET_SENSORS if target_indices[s] is not None}
    trues_dict = {s: [] for s in TARGET_SENSORS if target_indices[s] is not None}
    timestamps = []

    print("Running sequential inference (Extracting 4 sensors simultaneously)...")
    with torch.no_grad():
        for i in range(test_start_idx, test_start_idx + PREDICT_STEPS):
            if i >= total_len: break
            hist_data = data_values_norm[i - seq_len: i].T
            feat_temp = hist_data[np.newaxis, :, :]
            feat_zone = np.tile(zone_features[:, :, np.newaxis], (1, 1, seq_len))
            x_input = np.concatenate([feat_temp, feat_zone], axis=0)
            x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)

            _, sensor_feats = model.gwn(x_tensor, pre_adj_t)
            batch_sensor_coords = model.sensor_coords.unsqueeze(0)
            batch_t_zero = torch.zeros((1, len(sensor_ids), 1), device=device)
            interp_feats = model.interpolator(batch_sensor_coords, model.sensor_coords, sensor_feats)
            decoder_input = torch.cat([batch_sensor_coords, batch_t_zero, interp_feats], dim=-1)

            y_hybrid_norm = model.decoder(decoder_input).squeeze(-1)
            pred_all = y_hybrid_norm.cpu().numpy().flatten() * (temp_max - temp_min) + temp_min

            for s_id in preds_dict.keys():
                idx = target_indices[s_id]
                preds_dict[s_id].append(pred_all[idx])
                trues_dict[s_id].append(data_values_raw[i, idx])
            timestamps.append(full_df.index[i])

    # ================= 5. 画 Panel B - E (4个传感器排布) =================
    axes_positions = [gs[0, 1], gs[0, 2], gs[1, 1], gs[1, 2]]
    axes_obj_list = []

    for i, (s_id, pos) in enumerate(zip(TARGET_SENSORS, axes_positions)):
        ax = fig.add_subplot(pos)
        axes_obj_list.append(ax)

        if s_id in preds_dict:
            ax.plot(timestamps, trues_dict[s_id], label='Observations', color='#7f7f7f', linewidth=2.5, alpha=0.7)
            ax.plot(timestamps, preds_dict[s_id], label='ST-PIGFN', color='#d62728', linewidth=1.8, linestyle='--')

            ax.set_ylabel('Temperature (°C)', fontweight='bold', fontsize=AXES_LABEL_FONTSIZE)
            ax.grid(True, linestyle=':', alpha=0.7)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
                tick.set_fontsize(TICK_LABEL_FONTSIZE)
            for tick in ax.get_yticklabels():
                tick.set_fontsize(TICK_LABEL_FONTSIZE)

            ax.set_xlabel('Time', fontweight='bold', fontsize=AXES_LABEL_FONTSIZE, labelpad=5)

            title_text = PANEL_TITLES[i + 1]
            ax.text(0.5, LABEL_Y_OFFSET_B_E, title_text,
                    transform=ax.transAxes, ha='center', va='top',
                    fontweight='bold', fontsize=PANEL_LABEL_FONTSIZE)

            ax.legend(loc='lower left', fontsize=LEGEND_FONTSIZE, framealpha=0.8)

    # ================= 6. 保存图像与排版强制对齐 =================
    fig.canvas.draw()

    pos_a = ax1.get_position()
    pos_b = axes_obj_list[0].get_position()
    pos_d = axes_obj_list[2].get_position()

    base_y0 = pos_d.y0
    base_height = pos_b.y1 - pos_d.y0

    # 🚀 应用设置的上下移动偏移量 PANEL_A_BOX_Y_OFFSET
    ax1.set_position([pos_a.x0, base_y0 + PANEL_A_BOX_Y_OFFSET, pos_a.width, base_height])

    # 🚀 将标签 (a) 钉在原本固定的绝对坐标系上，解绑图框！
    # 这样，就算上面的框往下移动了，(a) 这个字依然保持在它该在的位置，和右边完美平齐。
    fig_x_a = pos_a.x0 + pos_a.width / 2.0
    fig_y_a = base_y0 + LABEL_Y_OFFSET_A * base_height

    fig.text(fig_x_a, fig_y_a, PANEL_TITLES[0],
             ha='center', va='top', fontweight='bold', fontsize=PANEL_LABEL_FONTSIZE)

    output_path = './' + doc + '/Figure_3_Composite_Model_Performance.png'
    plt.savefig(output_path, dpi=1000, bbox_inches='tight')
    print(f"✅ Figure 3 已生成并保存至: {output_path}")


if __name__ == "__main__":
    plot_figure_3()