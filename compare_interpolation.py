#!/usr/bin/env python3.8.10
# -*- coding: utf-8 -*-
"""
function description: 此文件用于
加载数据与模型：加载测试集数据和你训练好的 ST-PIGFN 模型。
设置对比基线：实现 IDW (反距离加权) 和 Kriging (克里金插值) 算法。
“遮挡-恢复”测试循环：
遍历测试集的时间点。在每个时间点，随机“遮挡”20%的传感器。
同时按全窟(Global)以及前室(0)、后室(1)、诵经道(2)分别计算并统计评价指标。
author: TangKan & AI Assistant
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time

# 引入项目模块
from LogicAlgorithm.DL_method.GWNPINN.model_1 import SpatioTemporalPINN

# 引入插值库
try:
    from pykrige.ok3d import OrdinaryKriging3D
except ImportError:
    print("错误: 未安装 pykrige。请运行 'pip install pykrige'。")
    exit()


# === 1. IDW 插值算法实现 ===
def idw_interpolation(train_coords, train_vals, query_coords, power=2):
    preds = []
    for q_coord in query_coords:
        dists = np.linalg.norm(train_coords - q_coord, axis=1)
        if np.min(dists) < 1e-6:
            idx = np.argmin(dists)
            preds.append(train_vals[idx])
            continue
        weights = 1.0 / (dists ** power)
        weights /= weights.sum()
        pred = np.dot(weights, train_vals)
        preds.append(pred)
    return np.array(preds)


# === 2. Kriging 插值算法实现 (3D 修正版) ===
def kriging_interpolation(train_coords, train_vals, query_coords):
    try:
        ok3d = OrdinaryKriging3D(
            train_coords[:, 0], train_coords[:, 1], train_coords[:, 2],
            train_vals, variogram_model='linear', verbose=False, enable_plotting=False
        )
        k3d_pred, _ = ok3d.execute('points', query_coords[:, 0], query_coords[:, 1], query_coords[:, 2])
        return k3d_pred
    except Exception as e:
        return np.full(len(query_coords), np.mean(train_vals))


# === 辅助函数 ===
def rebuild_scaler(params):
    scaler = MinMaxScaler()
    scaler.min_ = np.array(params['min_'])
    scaler.scale_ = np.array(params['scale_'])
    scaler.data_min_ = np.array(params['data_min_'])
    scaler.data_max_ = np.array(params['data_max_'])
    scaler.n_samples_seen_ = 100
    return scaler


def load_data_and_coords(data_path, coords_path, grotto_id):
    coords_df = pd.read_csv(coords_path)
    coords_df = coords_df[coords_df['grottoe_id'] == grotto_id]
    sensor_ids = coords_df['sensor_id'].tolist()
    coords_values = coords_df[['x', 'y', 'z']].values

    dfs = []
    print("Loading sensor data...")
    for sid in sensor_ids:
        f = os.path.join(data_path, f"{sid}.csv")
        if not os.path.exists(f): continue
        df = pd.read_csv(f, usecols=['time', 'air_temperature'])
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time').rename(columns={'air_temperature': sid})
        df = df[~df.index.duplicated(keep='first')]
        dfs.append(df)

    if not dfs:
        raise ValueError("No data found! Check paths.")

    full_df = pd.concat(dfs, axis=1).sort_index()
    full_df = full_df.interpolate(method='time').ffill().bfill()
    return full_df, coords_values, sensor_ids, np.array(coords_df['zone_id'].values)


def main():
    # ================= 配置区域 =================
    TARGET_GROTTO_ID = 10
    TEST_SAMPLE_STRIDE = 50
    MASK_RATIO = 0.2

    CHECKPOINT_PATH = f'./results_slices_20260406_02_有Q_！/grotto_{TARGET_GROTTO_ID}_model_checkpoint.pth'
    DATA_PATH = './data'
    COORDS_PATH = './data/_sensor_coords.csv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================= 1. 加载模型与环境 =================
    print("Loading Model...")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    except:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    scalers_params = checkpoint['scalers_params']
    temp_scaler = rebuild_scaler(scalers_params['temp_scaler'])
    temp_min = scalers_params['global_min']
    temp_max = scalers_params['global_max']
    graph_data = checkpoint['graph_data']
    adj_matrix = np.array(graph_data['adj_matrix'])

    config = checkpoint['config']
    train_info = checkpoint['train_info']
    seq_len = train_info['seq_len']

    model = SpatioTemporalPINN(
        num_nodes=config['num_nodes'], sensor_coords_norm=config['sensor_coords_norm'],
        in_dim=config['input_dim'], out_dim=config['out_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # ================= 2. 加载数据与初始化指标字典 =================
    full_df, coords_raw, sensor_ids, zone_ids = load_data_and_coords(DATA_PATH, COORDS_PATH, TARGET_GROTTO_ID)
    full_df_norm = pd.DataFrame(temp_scaler.transform(full_df.values), index=full_df.index, columns=full_df.columns)
    zone_one_hot = np.eye(3)[zone_ids].T

    total_len = len(full_df)
    test_start_idx = int(total_len * 0.9)
    data_values_norm = full_df_norm.values
    data_values_raw = full_df.values
    pre_adj_t = torch.tensor(adj_matrix, dtype=torch.float32, device=device)

    # 【新增核心逻辑】：构建按区域分类的评价指标字典
    ZONES = {
        'Global': 'All Sensors',
        0: 'Front Chamber (前室)',
        1: 'Rear Chamber (后室)',
        2: 'Chanting Passage (诵经道)'
    }
    METHODS = ['Ours (ST-PIGFN)', 'IDW', 'Kriging']

    metrics = {
        zone_key: {method: {'true': [], 'pred': []} for method in METHODS}
        for zone_key in ZONES.keys()
    }

    # ================= 3. 开始测试循环 =================
    print(f"\n🚀 开始分区域对比实验 (Mask Ratio: {MASK_RATIO})")
    start_time = time.time()
    count = 0

    for t in range(test_start_idx, total_len, TEST_SAMPLE_STRIDE):
        if t < seq_len: continue

        # --- A. 遮挡划分 ---
        num_sensors = len(sensor_ids)
        indices = np.arange(num_sensors)
        idx_train, idx_test = train_test_split(indices, test_size=MASK_RATIO, random_state=t)

        current_vals_raw = data_values_raw[t, :]
        coords_train, vals_train = coords_raw[idx_train], current_vals_raw[idx_train]
        coords_test, vals_test_true = coords_raw[idx_test], current_vals_raw[idx_test]

        # 获取当前测试节点所属的区域ID
        zones_test = zone_ids[idx_test]

        # --- B. 运行基线模型 ---
        pred_idw = idw_interpolation(coords_train, vals_train, coords_test, power=2)
        pred_kriging = kriging_interpolation(coords_train, vals_train, coords_test)

        # --- C. 运行本模型 ---
        hist_data = data_values_norm[t - seq_len: t].T
        feat_temp = hist_data[np.newaxis, :, :]
        feat_zone = np.tile(zone_one_hot[:, :, np.newaxis], (1, 1, seq_len))
        x_input = np.concatenate([feat_temp, feat_zone], axis=0)
        x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            _, sensor_feats = model.gwn(x_tensor, pre_adj_t)
            batch_sensor_coords = model.sensor_coords.unsqueeze(0)
            batch_t_zero = torch.zeros((1, len(sensor_ids), 1), device=device)
            interp_feats = model.interpolator(batch_sensor_coords, model.sensor_coords, sensor_feats)
            decoder_input = torch.cat([batch_sensor_coords, batch_t_zero, interp_feats], dim=-1)
            y_hybrid_norm = model.decoder(decoder_input).squeeze(-1)

            pred_all_model = y_hybrid_norm.cpu().numpy().flatten() * (temp_max - temp_min) + temp_min
            pred_model = pred_all_model[idx_test]

        # --- D. 按区域统计结果 ---
        if not np.isnan(pred_idw).any() and not np.isnan(pred_kriging).any():
            methods_preds = {
                'Ours (ST-PIGFN)': pred_model,
                'IDW': pred_idw,
                'Kriging': pred_kriging
            }

            for m_name, m_pred in methods_preds.items():
                # 1. 记录全局 (Global)
                metrics['Global'][m_name]['true'].extend(vals_test_true)
                metrics['Global'][m_name]['pred'].extend(m_pred)

                # 2. 按区域记录 (0, 1, 2)
                for z_key in [0, 1, 2]:
                    mask_z = (zones_test == z_key)
                    if np.any(mask_z):
                        metrics[z_key][m_name]['true'].extend(vals_test_true[mask_z])
                        metrics[z_key][m_name]['pred'].extend(m_pred[mask_z])

        count += 1
        if count % 10 == 0:
            print(f"Step {t}/{total_len} completed...")

    print(f"Comparison finished in {time.time() - start_time:.2f}s")

    # ================= 4. 打印格式化表格 =================
    print("\n" + "=" * 65)
    print("📊 空间插值方法分区域对比结果 (Test Set Average)")
    print("=" * 65)

    for z_key, z_name in ZONES.items():
        print(f"\n>>> 区域: {z_name} <<<")
        print(f"{'Method':<20} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10}")
        print("-" * 60)

        for method_name, data in metrics[z_key].items():
            if len(data['true']) == 0:
                continue

            y_true = np.array(data['true'])
            y_pred = np.array(data['pred'])

            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            print(f"{method_name:<20} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f}")

    print("\n" + "=" * 65)


if __name__ == '__main__':
    main()