#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于单独生成指定传感器的时间序列预测对比图 (Observations vs ST-PIGFN)
无论列表中有几个传感器，都会一张一张独立保存为高精度 SCI 图片。
author: TangKan
contact: 785455964@qq.com
IDE: PyCharm Community Edition 2024.2.2
time: 2026/4/20 13:14
version: V1.0
"""


import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 导入你原有的库
from predict_model import rebuild_scaler, load_merged_dataframe
from LogicAlgorithm.DL_method.GWNPINN.model_1 import SpatioTemporalPINN


def plot_individual_sensors():
    # ================= 1. 🔧 用户控制台 =================
    TARGET_GROTTO_ID = 10

    # 🚀 在这里填入你想要独立输出的传感器列表 (数量不限)
    # 'A33', 'C29', 'D27'
    TARGET_SENSORS = ['A37', 'A36', 'A35', 'A31', 'A30']

    PREDICT_STEPS = 864  # 测试步数 (3天)

    # 模型存放文件夹名称
    doc = 'results_slices_20260406_02_有Q_！'

    CHECKPOINT_PATH = './' + doc + f'/grotto_{TARGET_GROTTO_ID}_model_checkpoint.pth'
    DATA_PATH = './data'
    COORDS_PATH = './data/_sensor_coords.csv'

    # 建立一个文件夹专门存放这些单张图
    OUTPUT_DIR = './' + doc + '/Single_Sensor_Plots'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================= 2. SCI 画图全局风格设置 =================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    # ================= 3. 准备数据与模型加载 =================
    print(f"正在加载模型: {CHECKPOINT_PATH}")
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

    # 找到目标传感器在模型输出中的索引
    target_indices = {}
    valid_sensors = []
    for s_id in TARGET_SENSORS:
        if s_id in sensor_ids:
            target_indices[s_id] = sensor_ids.index(s_id)
            valid_sensors.append(s_id)
        else:
            print(f"⚠️ 警告: 传感器 {s_id} 不在模型的节点列表中，将跳过！")

    if not valid_sensors:
        print("❌ 错误：没有找到任何有效的传感器，程序退出。")
        return

    # 初始化模型
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

    # 存储预测和真实值
    preds_dict = {s: [] for s in valid_sensors}
    trues_dict = {s: [] for s in valid_sensors}
    timestamps = []

    print(f"正在进行连续时序推理... 将提取 {len(valid_sensors)} 个传感器数据。")
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

            for s_id in valid_sensors:
                idx = target_indices[s_id]
                preds_dict[s_id].append(pred_all[idx])
                trues_dict[s_id].append(data_values_raw[i, idx])

            timestamps.append(full_df.index[i])

    # ================= 4. 流水线单张出图 =================
    print("\n开始逐个生成并保存图片...")
    for s_id in valid_sensors:
        # 每画一张图都创建一个全新的干净画布
        fig, ax = plt.subplots(figsize=(10, 5))

        # 绘制真实值与预测值 (与你 Figure 3 保持完全一致的配色)
        ax.plot(timestamps, trues_dict[s_id], label='Observations', color='#7f7f7f', linewidth=2.5, alpha=0.7)
        ax.plot(timestamps, preds_dict[s_id], label='ST-PIGFN', color='#d62728', linewidth=2.0, linestyle='--')

        ax.set_ylabel('Temperature (°C)', fontweight='bold')
        ax.set_xlabel('Date & Time', fontweight='bold', labelpad=10)
        ax.set_title(f'Prediction Performance - Sensor {s_id}', fontweight='bold', pad=15)

        ax.grid(True, linestyle=':', alpha=0.7)

        # 时间轴格式化
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)

        # 图例固定在左下角
        ax.legend(loc='lower left', fontsize=12, framealpha=0.9)

        # 保存高精度图片
        output_file = os.path.join(OUTPUT_DIR, f"Prediction_Sensor_{s_id}.png")
        plt.savefig(output_file, dpi=1000, bbox_inches='tight')
        plt.close(fig)  # 必须 close 释放内存！

        print(f"✅ 成功生成: {output_file}")

    print("\n🎉 所有指定传感器的独立预测图已全部生成完毕！")


if __name__ == "__main__":
    plot_individual_sensors()
